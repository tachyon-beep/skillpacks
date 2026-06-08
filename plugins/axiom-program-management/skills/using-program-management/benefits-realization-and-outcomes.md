# Benefits Realization and Outcomes

**A program is not accountable for shipping the roadmap. It is accountable for the value the roadmap was supposed to produce — and those are different things that fail in different ways.** The central failure of program delivery is declaring success at the moment of delivery: the features are live, the burndown hit zero, the steering committee claps, the team disbands — and the metric the whole thing existed to move never moves. The work shipped; the value didn't land; and because "done" was defined as delivery, nobody noticed the gap until the next funding cycle asked what the investment returned. This sheet is about closing that gap: defining success as realized value, tracing every output to a benefit, naming someone accountable for realization *past* the point of delivery, and being willing to stop when the benefit doesn't materialize.

This is the pack's spine anti-pattern — **success defined as delivery, not realization** — and the program tier is where it does the most damage, because a program exists precisely to deliver an outcome no single project owns. If no one owns the outcome, the program is just a stack of projects with a shared status deck.

## The value chain: output → outcome → benefit → impact

Most delivery measurement stops at the first link. The discipline is to measure the whole chain and to know which link you are actually quoting when you claim success.

- **Output** — *what we built.* A shipped feature, a deployed service, a migrated dataset, a trained cohort. Outputs are fully within the team's control and easy to count, which is exactly why delivery gravitates to measuring them: "we shipped the new checkout flow" is verifiable on the day. An output is necessary for value and is never itself the value.
- **Outcome** — *the behaviour or state change that results from the output.* Users complete checkout instead of abandoning it; support agents resolve tickets without escalating; the reporting team self-serves instead of raising data requests. Outcomes are a change in what people *do* or what is *true*, caused by the output but not guaranteed by it. The output ships on a date; the outcome happens (or doesn't) over the following weeks, and only partly under the team's control — it depends on adoption, on whether the change was actually wanted, on the rest of the system.
- **Benefit** — *the measurable value that the outcome delivers.* Reduced revenue leakage from abandoned carts; lower cost-to-serve per ticket; analyst hours freed for higher-value work. A benefit is an outcome expressed in the currency the business case was written in (money, time, risk reduction, capacity). It is what justified the spend.
- **Impact** — *the longer-term strategic effect.* Market position, competitive moat, regulatory standing, organizational capability. Impact is slow, multi-causal, and rarely attributable to one program cleanly — which is why you track it but do not promise it as a program deliverable.

The relationship is causal and lossy: each arrow can fail. A great output can produce no outcome (nobody adopts it). A real outcome can produce no benefit (the behaviour changed but the value was cannibalized elsewhere). A benefit can produce no impact (it was too small to matter strategically). **Delivery that measures only outputs is flying blind on three of the four links** — it can confirm the thing was built and assert everything downstream as faith.

## The output–outcome gap

The output–outcome gap is the distance between "we shipped everything we said we would" and "the value we promised showed up." It is the most common program failure and the most invisible, because every local signal can be green while the gap is total: every sprint closed, every milestone hit, every ticket delivered, RAG green throughout — and the business case unrealized.

It is invisible for a structural reason. Outputs land *during* the program, when everyone is watching; benefits land *after*, when the program is winding down and attention has moved on. The instruments that ran hot during delivery (the board, the burndown, the standup) all measure outputs and go dark exactly when the benefit was supposed to arrive. Nobody is watching the link that fails.

Closing the gap is not a reporting fix; it is a definition fix. Success has to be defined as realization from the start — in the charter, in the business case, in the OKRs — so that "done" means *the metric moved*, not *the feature shipped*. Everything else in this sheet is machinery for holding that definition: mapping so outputs trace to benefits, ownership so someone is accountable when delivery ends, leading indicators so the gap is visible early, and kill criteria so a program that isn't closing the gap can be stopped.

## Benefits mapping: work backward from the benefit

A benefits map (or benefit dependency network) is built **backward**. You start from the desired benefit and ask what outcome would produce it; from that outcome, what capability or output would enable it; and only then what work to fund. This is the inverse of how roadmaps usually grow — forward, from "what can we build," which produces outputs in search of a justification.

Working backward enforces a discipline forward-planning can't: **every output must trace to a benefit, and any output that traces to nothing gets cut.** The map is a directed graph from benefits on the right to enabling outputs on the left, and an output with no path to a benefit is, by construction, waste — it consumed capacity and bought no value. Finding those orphans is the single highest-leverage thing a benefits map does.

A worked example, generic. Suppose the desired benefit is *reduce annual revenue lost to abandoned checkouts*:

```
BENEFIT                      OUTCOME                          ENABLING OUTPUT
(measurable value)           (behaviour/state change)         (what we build)

Reduce revenue lost     <--  More sessions complete      <--  One-page checkout
to abandoned                 checkout (fewer abandon          (collapse 4 steps → 1)
checkout                     at the payment step)
                                                          <--  Saved payment methods
                                                               (returning users skip entry)

                        <--  Fewer abandonments from     <--  Inline validation
                             form errors                      (catch errors before submit)

                                                          <--  Express wallet integration
                                                               (Apple/Google Pay)

                             [orphan check] ───────────────  "Redesign account
                                                               settings page"
                                                               → traces to NO outcome
                                                               on this benefit → CUT
                                                               (or justify under a
                                                               different benefit)
```

Read the map two ways. Left-to-right, it is a claim chain: *if* we build inline validation, *then* fewer people abandon on form errors, *then* we recover some of the lost revenue — and each arrow is a hypothesis you can later test against reality. Right-to-left, it is a justification audit: every benefit must have at least one outcome under it, every outcome at least one output, and every output must reach a benefit. The "redesign account settings page" item in the example is the kind of plausible-sounding work that survives forward planning and dies on a benefits map — it builds something, but it moves nothing this program is accountable for.

At small scale this is a whiteboard exercise done once and revisited at major decision points. At program scale, across many projects and a formal business case, the benefit dependency network becomes a maintained artifact with named owners on each benefit node and explicit measures on each — and in large, regulated, or public-funded programs, formal benefits management (a benefits register, profiles per benefit, periodic benefit reviews) is legitimate and often mandated, not bureaucracy. Match the rigor to the stakes: a single squad doesn't need a benefits register; a nine-figure transformation does.

## OKRs done well: the Objective is an outcome, the Key Results measure it

OKRs are the most common modern vehicle for outcome-orientation, and the most commonly corrupted. Done well they are a precise instrument for the output–outcome gap; done badly they are a task list wearing outcome clothing.

- The **Objective** is a qualitative *outcome* — a state or behaviour change, stated in plain language, ambitious and directional. "Make checkout something users complete instead of abandon."
- Each **Key Result** is a *measurable signal that the outcome happened* — a number that moves only if the behaviour actually changed. Not a deliverable. Not a task. A measure.

The dominant failure is **Key Results that are output/task checklists** — "ship one-page checkout," "integrate Apple Pay," "launch saved payment methods." These are things the team will *do*; they are roadmap items with a checkbox. You can complete every one of them and the Objective can entirely fail to materialize — which is the output–outcome gap reproduced *inside* the goal-setting framework that was supposed to prevent it. If your KRs are all things you control and can guarantee by working hard, they are outputs, and your OKRs are measuring delivery, not realization.

The test: a good KR can *fail even though you did all the work*, because it measures a result that depends on the world responding. That exposure to reality is the whole point.

| | Bad (output/task list) | Good (outcome measure) |
|---|---|---|
| **Objective** | "Launch the new checkout" | "Make checkout something users complete, not abandon" |
| **KR 1** | Ship one-page checkout flow | Reduce checkout abandonment rate from 70% → 55% |
| **KR 2** | Integrate Apple Pay and Google Pay | Increase checkout completion on mobile from 48% → 65% |
| **KR 3** | Launch saved payment methods | Cut median time-to-purchase for returning users from 90s → 40s |
| **What it measures** | That the team did the work | That the work produced the change |
| **Can it pass while value fails?** | Yes — ship all three, abandonment unchanged, "success" declared | No — these only move if behaviour actually changed |

The outputs in the bad column aren't wrong to do — they are exactly the enabling outputs from the benefits map. The error is *promoting them to the goal*. They belong on the roadmap as the bets; the KRs are how you'll know the bets paid off. Keep the outputs as the plan and the outcome measures as the OKR, and the framework does its job: it makes shipping necessary but not sufficient.

## Outcome ownership: a named owner accountable past delivery

A benefit needs a **named owner accountable for its realization** — and crucially, accountable *past the point of delivery*, because realization usually happens after the delivery team has moved on. This is the role the governance structure calls the Senior Responsible Owner (SRO) or equivalent accountable executive: the program is accountable for the outcome, and one named person is accountable for the program. Benefit ownership is not a parallel invention — it is that same accountability, extended forward in time to the moment the value is supposed to land.

The failure is structural and quiet. A program is funded, staffed, delivered, and disbanded on a delivery timeline — but the benefit it was funded for realizes on a *value* timeline that extends weeks or months past go-live. If ownership ends at delivery, the benefit enters a window where no one is watching for it and no one is accountable if it fails to appear. The value doesn't fail loudly; it fails by omission, because the org's attention and accountability both expired before the measurement date.

The discipline: name the benefit owner in the charter, give them the leading and lagging measures, and make benefit realization an explicit handover — the delivery team hands the *running* feature to the benefit owner, who remains accountable for the *value* until the realization measurement is in. In operational settings this often means the benefit owner is a business-side leader (the head of the function that will see the cost-to-serve drop), not the delivery lead — because they are the one still present, and still accountable, when the number is finally read.

## Leading benefit indicators: measure realization early

A benefit measured only at the end of the program is measured too late to do anything about. The lagging benefit number — annual revenue recovered, full-year cost-to-serve reduction — arrives after every decision that could have changed it has already been made. You need **leading indicators of realization**: early, partial proxies that tell you whether the benefit is *on track to land* while you can still act.

This is the same leading-vs-lagging discipline that status reporting uses to detect watermelon reports, applied to value instead of to delivery. A leading benefit indicator is a measure that (a) moves early, within weeks of the output landing, and (b) is causally upstream of the lagging benefit — so that movement in the proxy predicts movement in the benefit.

Worked example, on the checkout benefit:

```
LAGGING BENEFIT (end-of-program, too late to steer):
  Annual revenue recovered from reduced abandonment
  → measured at full-year close. By the time it's red, the program is over.

LEADING INDICATORS (visible in weeks, while you can still act):
  Week 1–2:  Checkout abandonment rate on the cohort exposed to the new flow
             (A/B: does the behaviour change at all?)
  Week 2–4:  Completion-rate delta between new flow and old flow
             (Is the outcome real, or did adoption stall?)
  Week 3–6:  Recovered-cart revenue run-rate, annualized from early weeks
             (Does the outcome convert to the benefit at the rate the
              business case assumed — or half of it?)
```

If the week-2 abandonment proxy doesn't move, the outcome isn't happening, and no amount of waiting for the annual number will change that — you have an early, actionable red while the program still has budget and a team. Leading indicators turn the benefit from a verdict delivered at the post-mortem into a signal you can steer by.

## The business case as a living document; disbenefits and cannibalization

A business case is usually written once, to secure funding, and then never opened again — a sales document, not a management instrument. Treated that way, its benefits are *asserted* at funding and never *validated* after. The discipline is to make the business case **living**: the benefits in it are tracked against actuals over the program's life and re-validated as the world changes, because the assumptions that justified the spend (market conditions, adoption rates, the size of the problem) drift, and a benefit that was real at funding can evaporate before delivery.

Living means tracking two things most business cases ignore:

- **Disbenefits** — the negative consequences of the change, honestly logged as the cost side of the value ledger. The new checkout that recovers carts may increase fraud-review load, or degrade an accessibility path, or raise support volume during the transition. A disbenefit is not a risk (it's not probabilistic — it's a known consequence you accept); it belongs in the benefits accounting so the *net* value is honest, not the gross.
- **Benefit cannibalization** — when a benefit is double-counted because two programs claim the same value, or when one program's benefit is another's disbenefit. If two initiatives both claim "reduce cost-to-serve by 10%," they cannot both be right against the same baseline; the second one is realizing a benefit the first already booked. At portfolio scale this is how a stack of individually-justified business cases sums to a promised value the organization can never actually realize.

## Kill and pivot criteria: define the stopping signal in advance

A program that cannot be stopped when its benefits fail to materialize is a sunk-cost machine: every month of spend becomes an argument for one more month, because stopping would "waste" what's been invested. The defence is to define, **in advance**, the benefit signals that justify stopping or pivoting — before the team is emotionally and reputationally committed, while the criteria can still be set honestly.

Kill/pivot criteria are stated against the leading indicators, not the lagging one (waiting for the lagging number defeats the purpose): *if the abandonment proxy hasn't moved by N points by week 6, we stop and reassess.* They turn the hardest decision in program management — admitting the bet isn't paying off — from a fraught judgment call made under sunk-cost pressure into a pre-agreed trigger that fires on data. A pivot criterion is the softer sibling: a signal that the outcome is real but the path is wrong, justifying a change of approach rather than a stop.

This is where the lean stance is sharpest and also where it is most often resisted: a program with no kill criteria is a program that has decided in advance it will never stop, regardless of evidence — which means it was never really a bet, just a commitment dressed as one. At larger and regulated scale the stage-gate review (a formal go/no-go at funded checkpoints) is the institutional form of the same discipline, and it earns its rigor; lean teams should run the same logic with lighter machinery.

## Anti-Patterns

1. **Success defined as delivery, not realization.** The roadmap ships in full, the business case never materializes, and the program disbands before anyone measures whether the value showed up — because "done" meant features delivered, not benefits realized. This is the pack's spine failure, and it survives precisely because every local delivery signal is green while the only signal that matters goes unmeasured. *Fix: define success in the charter as the benefit metric moving; treat shipped output as necessary but not sufficient, and hold the program open until realization is measured.*

2. **OKRs whose Key Results are task lists, not outcome measures.** "Ship feature X," "integrate Y," "launch Z" as KRs — deliverables the team controls and can guarantee by working hard, which means they measure effort, not effect. You can complete all of them and the Objective can entirely fail. *Fix: write KRs as outcome measures that can fail even when all the work is done ("reduce X from A% to B%"); keep the deliverables on the roadmap as the bets, not in the OKR as the goal.*

3. **No one owns the benefit past go-live.** Ownership ends when delivery ends, but realization happens weeks or months later — so the benefit enters a window with no accountable owner and fails silently by omission. *Fix: name a benefit owner (the SRO or a business-side accountable leader) in the charter, accountable past delivery, with an explicit handover of the running output and a committed realization measurement date.*

4. **Benefits asserted in the business case and never measured again.** The case is written to win funding and never reopened; its benefits are claims, not tracked actuals, and nobody notices when the assumptions that justified them drift. *Fix: make the business case living — track benefits against actuals on a cadence, re-validate assumptions, and log disbenefits and cannibalization so net value stays honest.*

5. **Vanity outcome metrics.** A number that looks like an outcome but isn't tied to real value — page views, logins, "engagement," features-adopted-once — chosen because it reliably goes up, not because it predicts the benefit. It produces the *feeling* of outcome measurement while measuring nothing the business case cares about. *Fix: require every outcome metric to trace to a benefit on the map; if moving the metric wouldn't move the benefit, it's vanity — replace it with a leading indicator that is causally upstream of the value.*

6. **A program with no kill criteria.** No pre-agreed signal that would justify stopping, so sunk cost runs the decision: every dollar spent becomes a reason to spend the next one, and the program continues regardless of whether the benefit is appearing. *Fix: define kill and pivot criteria in advance against the leading indicators ("if proxy hasn't moved N by week 6, stop and reassess"); at scale, formalize as stage-gate go/no-go reviews.*

## Cross-References

- [program-structure-and-governance.md](program-structure-and-governance.md) — the program is the structure accountable for the outcome that no single project owns; the SRO (or equivalent accountable executive) named there is the same role that owns benefits here, extended past delivery. Governance defines who decides; this sheet defines what they are accountable for realizing.
- [roadmapping-and-prioritization.md](roadmapping-and-prioritization.md) — once outputs trace to benefits on the map, sequence them by value: the benefit a piece of work serves is the input to cost-of-delay and WSJF prioritization, and outputs that trace to no benefit shouldn't be on the roadmap at all.
- [status-reporting-and-metrics.md](status-reporting-and-metrics.md) — outcome confidence is what RAG should track, not effort; the leading benefit indicators here are exactly the leading signals that sheet uses to carry bad news early and detect watermelon reports. Realization risk is a reportable status, not an end-of-program surprise.
- [scope-and-backlog-management.md](scope-and-backlog-management.md) — slice toward the outcome, not toward feature-completeness; the benefits map is the authority for cutting outputs that trace to no benefit, and the explicit-trade discipline there is how you defend the cut against "but it's nearly built."
- `/lyra-ux-designer` — product discovery and UX research *define* the desired outcomes (what behaviour change is worth pursuing and why users want it); this pack manages *realizing* them. The outcome on a benefits map is discovery's output; closing the gap to the benefit is this pack's job.
