# Delivery Cadence and Flow

**The job of delivery management is to make work *flow* and to make that flow *measurable*. Cadence is the rhythm you impose; flow is what actually happens; flow metrics are the instruments that tell you the difference between the two.** Most teams manage cadence (the sprint calendar, the standup) and never measure flow — so they feel busy, hit their ceremonies, and still cannot say why delivery is unpredictable.

This sheet covers choosing a cadence, limiting work in progress, and the four flow metrics that tell you the truth that velocity hides. It is the engine room of the pack: estimation forecasts *from* these metrics (`estimation-and-forecasting.md`), and status reports *on* them (`status-reporting-and-metrics.md`).

## Cadence: iteration, flow, or hybrid

Cadence is the heartbeat of delivery. There are three viable shapes, and the choice is driven by how predictable the work is and how much synchronization the team needs — not by fashion.

**Iteration-based (Scrum-style sprints).** Work is batched into fixed timeboxes (commonly two weeks). The timebox forces a planning rhythm, a review rhythm, and a natural inspection point. It suits work that benefits from a regular commitment-and-review beat: a team that needs stakeholders to see increments on a predictable schedule, or that uses the planning ceremony to force prioritization it would otherwise avoid.

The cost of iterations is the **batch boundary**. Work that doesn't fit the timebox gets split awkwardly or carried over; the sprint boundary becomes an artificial deadline that encourages cutting corners near the end and starting slow at the beginning (the "sprint sawtooth"). Carryover is the diagnostic: a team that carries items over most sprints is running a kanban system wearing a Scrum costume, and should either fix its slicing or drop the timebox.

**Flow-based (kanban).** Work is pulled continuously as capacity frees up, with no timebox. There is no sprint commitment; there is a WIP limit and a pull signal. Flow suits work that arrives unpredictably (operational/support work, incident-driven work) or work whose items vary wildly in size, where forcing a timebox adds ceremony without adding value. Flow systems are managed by WIP limits and flow metrics rather than by sprint commitments.

**Hybrid (cadence for sync, flow for work).** The common mature endpoint: keep a regular cadence for the *synchronization* events that need it — a planning checkpoint, a stakeholder review, a retro — but manage the *work* as a continuous pull system with WIP limits. You get the rhythm where rhythm helps (alignment, inspection) and continuous flow where batching hurts (the actual delivery). Most teams that "do Scrum well" for years are really running a hybrid and have stopped treating the sprint commitment as sacred.

**Decision heuristic:**

| Condition | Lean toward |
|-----------|-------------|
| Work arrives in a planned backlog; stakeholders want predictable increments | Iteration |
| Work arrives unpredictably (ops, support, incidents) | Flow |
| Item sizes vary by more than ~5× | Flow or hybrid (timebox fights the variance) |
| Carryover happens most sprints | Flow or hybrid (the timebox isn't fitting) |
| Multiple teams must synchronize | Cadence for the sync points, flow underneath |

The cadence is not the management system. The management system is WIP limits and flow metrics — those work under any cadence.

## Work in progress is the master variable

The single highest-leverage control in delivery is the **WIP limit**: a cap on how many items the team is working on at once. It is counterintuitive because limiting WIP *feels* like leaving capacity on the table. It is the opposite: unlimited WIP is the most common cause of unpredictable delivery dates.

The mechanism is **Little's Law**, which holds for any stable queue:

```
Average Cycle Time = Average WIP / Average Throughput
```

Where **cycle time** is how long an item takes from start to done, **WIP** is the number of items in progress, and **throughput** is items completed per unit time. The law is an identity, not a model — it is always true on average for a stable system.

Read it as a manager: **for a given throughput, cycle time rises linearly with WIP.** If a team can finish 5 items a week (throughput) and is working on 10 items at once (WIP), the average item takes 2 weeks (cycle time). Double the WIP to 20 with the same throughput and cycle time doubles to 4 weeks — the team is no faster, but every item now takes twice as long to land, and the delivery date for any given item is twice as far out and twice as uncertain.

High WIP causes:

- **Longer cycle times** — directly, by Little's Law.
- **More context-switching** — each person juggling more items pays the switch tax, which *lowers* throughput, making the cycle-time penalty worse than linear.
- **Hidden bottlenecks** — when everything is started, the constraint is invisible; the bottleneck only reveals itself when WIP is capped and work piles up *in front of* it.
- **Late risk discovery** — an item barely started looks the same as an item nearly done on a board with no WIP discipline; problems surface late.

**Setting WIP limits.** Start near current throughput and tighten. A common starting point is roughly one item per person or slightly fewer (forcing some collaboration / swarming), then lower the limit until flow visibly improves and the bottleneck becomes obvious. The limit is correct when reducing it further starts starving the team and raising it lengthens cycle time without raising throughput. WIP limits are per-column on a kanban board (limit the *bottleneck* column hardest), not just a global cap.

**The swarming corollary.** When WIP is limited and an item blocks, the right move is for the team to swarm the blocker rather than start new work. "Stop starting, start finishing" is the slogan; the WIP limit is what makes it enforceable rather than aspirational.

## The four flow metrics

These four metrics measure delivery in units of reality. They are gathered from the board (timestamps on state transitions), not from estimates, which is exactly why they resist gaming.

**1. Cycle time** — elapsed time from when work *starts* on an item to when it is *done*. Measured per item, reported as a distribution (median and a high percentile, e.g. 85th), never as a bare average — cycle-time distributions are right-skewed, so the average is misleading and the percentile is what you forecast and promise against. "85% of our items finish within 9 days" is a sentence you can build a service-level expectation on. The average alone is not.

**2. Throughput** — number of items *completed* per unit time (per week is common). This is the team's actual delivery rate. It is the input to forecasting (`estimation-and-forecasting.md`): future delivery is projected from the distribution of past throughput, not from summed estimates. Count *items*, not points — items are countable in reality and throughput-based forecasting works on item counts directly.

**3. Lead time** — elapsed time from when an item is *requested* (enters the backlog as a commitment) to when it is *delivered*. Lead time ≥ cycle time always, because lead time includes the queue wait before work starts. The gap between lead time and cycle time is *queue time* — time the customer waits while the item sits in the backlog. Stakeholders experience lead time; the team often only manages cycle time, which is why "it only took us three days to build" coexists with "we asked for that two months ago." (Terminology varies between sources; what matters is that you measure both the *waiting* and the *working* and know which one you're quoting.)

**4. Work in progress** — the count of items currently started-but-not-done. Already covered as the control variable; as a metric, it is watched on a Cumulative Flow Diagram to spot WIP creeping up (the band widening) before cycle time visibly degrades.

**Flow efficiency** is the derived metric that exposes waste:

```
Flow Efficiency = (Active Time / Total Cycle Time) × 100%
```

Active time is time the item is genuinely being worked; the remainder is blocked/waiting time. Most unoptimized knowledge-work systems run at **15–25% flow efficiency** — meaning an item spends three-quarters or more of its cycle time waiting, not being worked. This is the number that reframes "we need to work faster" into "we need to stop our work from waiting." You almost never improve delivery by making active time shorter; you improve it by attacking the 75% of waiting — handoffs, blocks, queue depth, and dependency stalls (`dependencies-and-coordination.md`).

## The Cumulative Flow Diagram

The CFD is the one chart that shows all of this at once. It stacks, over time, the count of items in each board state (backlog, in progress, done, and any columns between). Reading it:

- **The width of the "in progress" band = WIP.** A widening band means WIP is growing — the early warning that cycle time is about to degrade.
- **The horizontal distance between the bottom of the "in progress" band and the "done" line ≈ cycle time.**
- **The vertical gap between arrival (top) and departure (bottom) lines = WIP at that moment.**
- **A flattening "done" line = throughput dropping.** A flat done line with a widening in-progress band is the classic "everyone's busy, nothing's shipping" signature.
- **A band that balloons in one column** points straight at the bottleneck.

A CFD is worth more than a burndown because it shows the *system*, not just one sprint's remaining work. Burndown answers "are we on track this sprint"; the CFD answers "is our delivery system healthy."

## Anti-Patterns

1. **Velocity as a productivity or comparison metric.** Story-point velocity is a team-local currency: 30 points on one team is not comparable to 30 on another, and "increase velocity" directly incentivizes point inflation, which makes the number rise while delivery does not. Velocity is, at best, one team's rough capacity-planning input within one stable team. Throughput (items/week, measured in reality) is what you forecast and report against. *Fix: track throughput and cycle time; if you keep velocity, use it only for that team's own capacity planning and never to compare teams or as a performance target.*

2. **Unlimited WIP / "we're all 100% utilized."** Full utilization of every person maximizes WIP and, by Little's Law, maximizes cycle time and unpredictability. A delivery system run at 100% utilization has the queue behavior of a freeway at 100% capacity: gridlock. *Fix: set WIP limits below comfortable, watch cycle time fall, accept visible slack as the price of flow.*

3. **Managing the sprint instead of the flow.** Treating the two-week burndown as the management instrument while ignoring cycle time and throughput. Burndown tells you about one batch; it is blind to the systemic WIP and queue problems that actually drive your dates. *Fix: add a CFD and cycle-time/throughput tracking; keep burndown only if the team finds it useful within a sprint.*

4. **Averages instead of distributions for cycle time.** Quoting "average cycle time is 5 days" when the distribution is right-skewed and the 85th percentile is 14 days. Commitments built on the average are wrong more than half the time on the items that matter. *Fix: report and promise against a high percentile (85th/95th), not the mean.*

5. **Chronic carryover treated as normal.** Carrying items over most sprints while still calling the sprint a "commitment." The commitment is fiction and everyone knows it, which corrodes the value of every other commitment. *Fix: either fix slicing so items fit (`scope-and-backlog-management.md`) or drop the timebox and run flow.*

6. **Optimizing active time when the problem is wait time.** Pushing the team to code faster when flow efficiency is 18% — i.e., when 82% of cycle time is waiting. Effort spent on the 18% can't move the date much; the leverage is entirely in the 82%. *Fix: measure flow efficiency; attack blocks, handoffs, and queues, not keystrokes.*

## Cross-References

- `estimation-and-forecasting.md` — consumes throughput and cycle-time distributions to forecast delivery dates probabilistically; this sheet produces the data that sheet forecasts from.
- `status-reporting-and-metrics.md` — reports these flow metrics upward as honest, gaming-resistant indicators; the watermelon-detection there leans on cycle time and throughput as leading signals.
- `scope-and-backlog-management.md` — slicing work so items are small and uniform enough to flow; chronic carryover is usually a slicing problem.
- `dependencies-and-coordination.md` — blocked/waiting time (the enemy of flow efficiency) is largely dependency-driven; that sheet covers owning the seams that cause the waits.
- `capacity-and-resource-flow.md` — at program scale, the WIP and flow principles here apply to the team-of-teams; that sheet covers why stable teams outperform resource-leveled pools.
- `/axiom-planning` — once the top backlog item is selected here, hand it to `/axiom-planning` to produce the executable implementation plan; this sheet manages the flow of work, not the plan for any one item.
