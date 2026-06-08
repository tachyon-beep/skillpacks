# Estimation and Forecasting

**An estimate is not a forecast. An estimate is a guess about one item's size; a forecast is a probabilistic statement about when a body of work lands. The dominant failure of delivery is to sum point estimates into a single committed date — which discards every bit of variance in the estimates and produces a number whose hidden hit-rate is ~50% or worse.** This sheet is the antidote: it shows how to forecast "when will it be done?" defensibly, from the distribution of historical throughput rather than from a column of summed guesses, and how to communicate the answer as a range with a confidence level instead of a date wearing a suit.

Forecasting *consumes* the flow data that `delivery-cadence-and-flow.md` produces — throughput per week and cycle-time distributions, measured in reality off the board. This sheet does not re-derive Little's Law or re-litigate velocity-vs-throughput; it assumes you have a throughput record (or shows you how to start one fast) and turns it into a date you can stand behind.

## Estimate, target, commitment — the root distinction

The single most expensive confusion in delivery is conflating three different things that all look like a number with a date attached. After McConnell (*Software Estimation: Demystifying the Black Art*), keep them rigorously separate:

| Term | What it is | Whose statement | Honest form |
|------|-----------|-----------------|-------------|
| **Estimate** | A probabilistic assessment of how big/long something is | The team's analytical judgment | "Most likely 6–10 weeks; P85 is 10" |
| **Target** | A business *desire* — a date someone wants to hit | The sponsor's wish | "We want it by end of Q3" |
| **Commitment** | A *promise* to deliver defined scope by a date | A negotiated agreement | "We commit to the P85 scope by Sep 30" |

An estimate is an input. A target is a constraint. A commitment is a decision made *with knowledge of both* — and a commitment should almost never be set equal to a raw target, nor to a P50 estimate. The disease is the sentence "the estimate is end of Q3," which silently fuses all three: a desire (the target) has been laundered into an analytical claim (the estimate) and hardened into a promise (the commitment) without anyone choosing a confidence level. Every defensible forecast keeps the three visibly distinct: here is what the data says (estimate/forecast), here is what you want (target), here is what we will promise and at what confidence (commitment).

## Relative estimation: what it is good for, and what it is not

Relative estimation sizes items *against each other* rather than in absolute time — story points (often a Fibonacci-ish scale: 1, 2, 3, 5, 8, 13) or t-shirt sizes (S/M/L/XL). The team asks "is this one bigger or smaller than that one, and roughly by how much?" rather than "how many hours is this?"

**Why relative beats absolute for sizing.** Humans are demonstrably bad at absolute time estimation (we anchor, we forget overhead, we ignore the queue) but considerably better at *comparison* — "this is about twice that" is a judgment people make reliably. Relative sizing also strips out the individual: a point is a property of the work, not of who happens to pick it up, so the estimate survives the work moving between people. And it is fast — a team can size a backlog in minutes by comparison where absolute estimation would take hours of false precision.

**What relative estimation is genuinely useful for:** a fast capacity-planning input *within one stable team*. A team that knows it completes ~40 points in a typical two-week window can use that to sketch how much of a sized backlog fits in a quarter — as a rough planning input, refined by forecasting.

**What it must never be used for:**

- **Cross-team comparison.** Points are a team-local currency. 30 points on one team is not 30 on another; the scales were never calibrated against each other and never can be. (`delivery-cadence-and-flow.md` covers why velocity is non-comparable in full.)
- **A productivity metric.** "Increase velocity" directly incentivizes point inflation — the number rises while delivery does not. The moment points become a target they stop measuring anything (Goodhart).
- **A forecast.** This is the central trap of this sheet. Points sized for relative capacity-planning are not a probability distribution; summing them and dividing by velocity to get a date discards all variance (see the next two sections). Sizing is not forecasting.

## Right-sizing and the #NoEstimates-aware move: count, don't sum

There is a quieter, often-better path: **right-size items to roughly uniform smallness and then count them instead of estimating them.** If you slice the backlog (see `scope-and-backlog-management.md`) so most items are about the same size — "would this take more than a few days? then split it" — then the *number of items* becomes a meaningful forecasting unit on its own. You stop asking "how many points?" and start asking "how many items, and how fast do we finish items?"

The empirical finding behind the #NoEstimates conversation is blunt and worth internalizing: **throughput of item-count often forecasts as well as or better than point-summing — because item counts are real and points are not.** A completed item is an observed fact on the board with a timestamp. A point is a pre-hoc judgment that may have been wrong. When you forecast from counts you forecast from reality; when you forecast from summed points you forecast from a pile of guesses and inherit all their error. Right-sizing also kills the long-tail item — the "13-pointer" that is really three undiscovered stories — by forcing it to be split before it enters the flow.

This is not anti-estimation dogma. Sizing still helps a team *slice* and *sequence*, and in some contexts (a fixed-bid proposal, a regulated milestone) you genuinely need a defensible up-front number before any throughput exists — that is the cold-start and reference-class case below. The point is narrower: **once you have a throughput record, forecast from item-count throughput, not from re-summed points.**

## Throughput-based forecasting

The base method needs no simulation. Take your historical **weekly throughput** — completed items per week, straight off the board — over a representative recent window. Suppose the last ten weeks were:

```
4, 6, 3, 5, 7, 2, 5, 4, 6, 3   →   total 45 items over 10 weeks
```

For a backlog of **N = 30** items, the naive answer is `30 ÷ (45/10) = 30 ÷ 4.5 ≈ 6.7 weeks`. Hold onto that number — it is the average-based point answer, and the next section shows it is mathematically the *coin-flip* date, not a safe one.

The honest version uses the **distribution**, not the average. The slowest weeks (2, 3) and fastest (7) bracket the range. A simple distributional forecast: a *pessimistic* line at the low-throughput end (`30 ÷ 3 = 10 weeks`), a *middle* at the average (`≈ 7 weeks`), an *optimistic* at the high end (`30 ÷ 5.5 ≈ 5.5 weeks`). That already gives you a range to talk in. But it treats the weeks as independent extremes rather than letting them combine the way reality does — a fast week followed by three slow ones, an outage, a holiday. To capture how the weeks actually *compound*, you sample them. That is Monte-Carlo.

## Monte-Carlo forecasting (mandatory mechanics)

Monte-Carlo forecasting answers "when?" by *replaying your own history thousands of times*. It is an algorithm a reader can implement in an afternoon; here it is precisely.

**Variant A — "how long will N backlog items take?"**

1. Take your historical **weekly-throughput record** — the list of items-completed-per-week, e.g. `[4, 6, 3, 5, 7, 2, 5, 4, 6, 3]`. **Include zero-throughput weeks** (holidays, all-hands-blocked weeks) in the record — dropping them biases the forecast optimistic, because those weeks really do happen and really do delay delivery.
2. Run **10,000 simulated trials**. In each trial:
   - Set `items_done = 0`, `weeks = 0`.
   - Repeat: sample one week's throughput **at random, with replacement**, from the historical record; add it to `items_done`; add 1 to `weeks`.
   - Stop when `items_done ≥ N`. The trial's result is `weeks`.
3. Collect all 10,000 `weeks` results into a distribution.
4. Read percentiles off the sorted distribution:
   - **P50** — the 5,000th value — your **coin-flip date**: 50% of trials finished by then, 50% did not.
   - **P85** — the 8,500th value — your **85%-confidence date**: a good default for an external commitment.
   - **P95** — the 9,500th value — for **high-stakes** commitments (regulatory deadlines, contractual penalties).

**A worked walkthrough with tiny numbers.** Take a short history `[2, 4, 3, 5, 4, 3, 6, 3]` (mean = 3.75) and a backlog **N = 30**. Hand-simulate three trials, sampling weeks with replacement:

| Trial | Sampled weekly throughputs (cumulative) | Weeks to reach 30 |
|-------|------------------------------------------|-------------------|
| 1 | 6, 4, 5, 3, 5, 4, 3 → 30 | **7** |
| 2 | 3, 2, 4, 3, 3, 5, 4, 3, 3 → 30 | **9** |
| 3 | 4, 3, 3, 4, 2, 6, 3, 5 → 30 | **8** |

Three trials already spread across 7–9 weeks. Run 10,000 and the spread fills in. For a history like this the distribution typically lands with **P50 ≈ 8 weeks, P85 ≈ 9 weeks, P95 ≈ 10 weeks**. Now compare to the naive average answer: `30 ÷ 3.75 = 8 weeks`. **The single number a manager would quote by dividing the backlog by average throughput lands at roughly P50 — the coin-flip date.** That is not a coincidence: `N ÷ average-throughput` approximates the *mean* completion time, and because the completion-time distribution is right-skewed the mean sits slightly *above* the median — so the naive number lands about at P50, often a hair to the safe side of it, never the comfortable margin it is mistaken for. The thesis holds either way: summing and dividing does not buy a *safe* date, it buys a date you will *miss roughly half the time*. The P85 (9 weeks) is what an honest commitment uses — over a week of buffer the naive method would have promised away with a straight face, and the P95 (10 weeks) is 25% further out still for when the downside is expensive.

**Variant B — "how many items can we deliver by date D?"** Invert the loop:

1. Compute the number of weeks until D (say **8 weeks**).
2. In each of 10,000 trials, sample 8 weekly-throughput values with replacement and **sum** them — that trial's result is "items delivered in 8 weeks."
3. Collect the distribution of item-counts and read percentiles — **but the direction inverts.** Here, *higher confidence means FEWER items.* You are 50% likely to deliver *at least* the P50 count and 85% likely to deliver *at least* the P15 count. So for a confident "we will deliver at least M by D" statement you quote a **low** percentile of the item-count distribution (P15 for ~85% confidence), not a high one.

The two variants invert because "more weeks" makes a deadline *easier* to hit while "more items" makes a scope target *harder* to hit. Keep the direction straight: for **time-to-finish-N**, high confidence = a **later** date (P85 > P50); for **items-by-D**, high confidence = a **smaller** committed scope (P15 < P50). Flipping this sign is the classic Monte-Carlo blunder.

**Why this beats a single estimate.** A point estimate gives you one number and hides the risk. Monte-Carlo hands you the *entire risk distribution* — you can see how steep the tail is, choose a confidence level appropriate to the stakes, and show the sponsor exactly what they are buying when they ask for an earlier date (they are buying a lower probability of hitting it). It also needs no estimation at all beyond counting items: it runs on throughput, which is already measured. The only inputs are the backlog size and the history.

**Assumptions to state honestly.** Monte-Carlo assumes the future resembles the sampled past: roughly stable team, roughly stable system, roughly fixed scope. If the team is changing size, the process is being reorganized, or scope is growing, the forecast degrades — sample from a window that reflects current reality, re-run on a cadence, and treat a forecast as a living number, not a one-time pronouncement. Scope change in particular *moves the forecast*: adding items to N pushes every percentile out, which is exactly why scope changes must be visible trades (`scope-and-backlog-management.md`).

## The cone of uncertainty and reference-class forecasting

Early estimates are uncertain by a *known, wide* factor, and the factor narrows as the work is understood. This is McConnell's **cone of uncertainty**: at the earliest "initial concept" stage, a realistic estimate spans roughly **0.25× to 4×** the eventual actual — a 16× spread between the optimistic and pessimistic ends. As requirements firm up the cone narrows; by the time requirements are complete it is closer to **0.8× to 1.25×**.

| Stage | Realistic estimate range (× actual) |
|-------|-------------------------------------|
| Initial concept | 0.25× – 4× |
| Requirements complete | 0.8× – 1.25× |

Two consequences. First, **an early single-point date is not just risky, it is dishonest about a factor you can name** — quoting "12 weeks" at concept stage when the honest statement is "somewhere between 3 and 48 weeks, and we'll narrow that as we learn." Second, the cone narrows *only as real uncertainty is resolved* — you cannot narrow it by re-estimating harder. Decomposing the work into ever-finer tasks and summing them does **not** correct the bias; it usually amplifies it, because bottom-up decomposition systematically *omits* the tasks no one thought of and compounds individual optimism across every line.

**Reference-class forecasting** corrects optimism bias better than bottom-up decomposition. Instead of building the estimate up from the parts of *this* effort (the inside view), you compare *this whole effort* to a class of *similar past efforts* and ask how long those actually took (the outside view). "Initiatives of about this shape, in this organization, have historically taken 5–9 months — what makes us think this one is different?" The outside view automatically includes the surprises, the integration pain, and the unplanned work, because those things happened in the reference class too. Bottom-up decomposition is seductive because it feels rigorous; the reference class is *accurate* because it is built from outcomes, not intentions.

## Cold start: forecasting with no history

The methods above need a throughput record. When you have none — a brand-new team, a greenfield initiative — do not retreat to summing estimates. Build a forecast anyway, and improve it fast:

1. **Borrow a reference class.** Use the outside view: similar past efforts (yours or industry), a comparable team's throughput, a sibling project's cycle times. An imported reference class with a wide interval beats a precise-looking bottom-up number built from nothing.
2. **Start small batches immediately to build a throughput record fast.** Right-size the first items hard and get a handful to *done* — even two or three weeks of real throughput is enough to seed a Monte-Carlo run. The fastest way to a defensible forecast is a few completed items, not a longer planning meeting.
3. **Forecast-and-correct on a short loop.** Re-run the forecast every week or two as throughput accrues. The forecast's job at cold start is not to be right; it is to *converge*, visibly, as data arrives.
4. **Widen the interval to match your ignorance.** With little history, quote a deliberately wide range and a lower confidence ("our current best guess is a P50 of ~Q3, but with two weeks of data the interval is wide and we'll tighten it"). Honesty about width *is* the rigor here.

## Communicating a forecast

A forecast is **always a range with a confidence level**, never a bare single date. Two equally good forms:

- A **confidence statement**: "We are 85% confident this completes by the end of Q3."
- A **percentile pair**: "P50 is mid-August; P85 is mid-September." The pair is more honest still, because it shows the sponsor the cost of the earlier date — the gap between P50 and P85 *is* the risk they would be absorbing by promising the earlier one.

**When the sponsor wants one number.** They often will — "just give me a date." Do not refuse and do not silently hand over the P50 (the coin-flip). Give the **P85 and name it as such**: "If you need a single date to commit to externally, use mid-September — that's the date we're 85% confident of. I can give you an earlier one, but every week I pull it in lowers the odds; mid-August is a coin flip." This converts a demand for false certainty into an explicit, sponsor-owned **risk decision**: they choose the confidence level, they see what they are trading. That is the difference between forecasting and fortune-telling.

Never pad the number secretly to feel safe. Padding hides the confidence level inside a fudge factor no one can interrogate; the P85 *is* the principled, transparent version of the safety margin — it states the confidence openly and is reproducible from the data.

## Anti-Patterns

1. **A committed date built by summing point estimates.** The pack's spine anti-pattern. Estimates are not probabilities; summing them and quoting the total discards every bit of variance and yields a number whose hit-rate is ~50% at best — and, as the walkthrough shows, the naive `N ÷ average throughput` lands at roughly P50 (it approximates the mean completion time, which for a right-skewed distribution sits just above the median), the date you miss about half the time. *Fix: forecast from the distribution of historical throughput (Monte-Carlo), commit at P85, and name the confidence level.*

2. **A single-point date with no confidence interval.** Any date quoted without a confidence attached is undefined — it could be a P50 or a P95 and the listener cannot tell, so they assume certainty that does not exist. *Fix: quote a range and a confidence (P50/P85 pair, or "85% confident by D"); a date with no interval is not a forecast.*

3. **Padding estimates secretly instead of stating a confidence level openly.** Adding an invisible fudge factor "to be safe" hides the real distribution inside a number no one can interrogate, and the pad is usually wrong in both directions. *Fix: replace the secret pad with an explicit percentile — the P85 is the transparent, reproducible version of the safety margin.*

4. **Treating velocity as a forecast.** Multiplying average velocity by remaining sprints to produce a date treats a noisy team-local capacity figure as a deterministic rate. It is the point-summing fallacy in story-point clothing. *Fix: forecast from item-count throughput with Monte-Carlo; use velocity only for one team's rough capacity sketch, never as the date.*

5. **Conflating estimate, target, and commitment.** Laundering a business desire (target) into an analytical claim (estimate) and hardening it into a promise (commitment) with no confidence level chosen — the sentence "the estimate is end of Q3" doing all three jobs at once. *Fix: keep the three visibly distinct; a commitment is a deliberate decision made at a chosen confidence, not a target relabeled.*

6. **Re-estimating endlessly for "more accurate" point estimates.** Burning planning cycles decomposing work ever-finer and re-summing, believing precision will emerge — when bottom-up decomposition amplifies optimism bias (it omits the unforeseen tasks) and never narrows the cone, which only real resolved uncertainty narrows. *Fix: stop re-estimating; forecast from throughput and reference classes, and let the cone narrow as the work is actually understood, not as it is re-decomposed.*

## Where formal statistical process management lives

This sheet is the **operational** treatment of forecasting: percentiles off a throughput distribution, Monte-Carlo over a date range, a P85 commitment. That is the right rigor for most delivery. In a **regulated or formally process-managed context** — where a program operates at CMMI Level 4 quantitative management, or a contract/auditor requires demonstrable statistical control of the process, not just a confidence interval on the outcome — the discipline shifts to **statistical process control**: control charts with computed upper/lower control limits, special-cause-vs-common-cause analysis, and process-capability baselines. That treatment lives in `/axiom-sdlc-engineering` (quantitative-management). The boundary is load-bearing: **forecast-from-throughput is the operational version; SPC with control limits is the formal-context version.** When a program needs to *prove the process is in statistical control* (not merely *forecast the date*), route to that pack; this sheet gets you a defensible date, that pack gets you a defensible *process*.

## Cross-References

- `delivery-cadence-and-flow.md` — produces the throughput and cycle-time distributions this sheet forecasts *from*; forecasting is the consumer, that sheet is the source of the data. Read it first if you have no throughput record.
- `scope-and-backlog-management.md` — a forecast assumes roughly-fixed scope; every added item pushes all percentiles out, which is exactly why scope change must be a visible, traded decision, not accretion.
- `status-reporting-and-metrics.md` — report the forecast as a range with a confidence level, never a bare date; the honest-RAG discipline there consumes the P50/P85 produced here.
- `roadmapping-and-prioritization.md` — program-scale sequencing (now/next/later, WSJF, cost of delay) uses these forecasts to order work by value against defensible delivery windows.
- `/axiom-sdlc-engineering` (quantitative-management) — the formal, regulated-context treatment: statistical process control, control limits, and process-capability baselines, for when a program must prove the *process* is in control, not just forecast the date. A load-bearing handoff, not a footnote.
