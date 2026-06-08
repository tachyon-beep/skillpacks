# Roadmapping and Prioritization

**A roadmap is a statement of intent about outcomes over time, and a prioritization is a defensible ordering of work by the economic value of doing it sooner. The recurring failure is to confuse the two with their counterfeits: a roadmap that is really a date-pinned feature Gantt presented as a promise, and a prioritization that is really the order in which people asked.** Both counterfeits feel like control and provide none. The roadmap communicates direction honestly under uncertainty; the prioritization decides what flows next; and the only place a *dated commitment* legitimately comes from is a forecast (`estimation-and-forecasting.md`) — never the roadmap itself.

This is a program-tier sheet because prioritization is where governance earns its keep: at single-team scale a simple cost-of-delay ordering suffices, but across a portfolio of competing themes the decision needs an explicit economic model and an owner. This sheet gives both — the honest roadmap shape, and the arithmetic that ranks work by value rather than volume.

## The roadmap is outcomes over time, not features over dates

A roadmap answers "where are we heading and roughly in what order," not "which feature ships on which date." The instinct to pin features to dates produces a document that looks authoritative and is structurally dishonest: under genuine uncertainty, a date-pinned feature roadmap presented as a *commitment* is a lie, because the dates were never forecasts — they were wishes laid out on a calendar. When (not if) the work runs differently than drawn, every downstream consumer who treated the picture as a promise is now managing a broken commitment that was never real.

The discipline is to separate the two artifacts that a Gantt-style roadmap fuses:

- The **roadmap** carries *intent and sequence* — what outcomes we are pursuing, in what rough order, with confidence that decreases as you look further out. It is allowed to be imprecise about timing because that imprecision is *honest*.
- The **forecast** carries *dated commitments* — and it is built from historical throughput and cycle-time distributions, expressed as a date *range* with a confidence level, not a single line on a chart. When a stakeholder needs a date, you hand them a forecast (`estimation-and-forecasting.md`), not a roadmap cell. The roadmap says "checkout improvements are the Now theme"; the forecast says "the first checkout slice lands in weeks 4–7 at 85% confidence."

Keeping these separate is what lets the roadmap stay stable while the forecast updates weekly. Fuse them and you get the worst of both: a roadmap that must be redrawn every time reality moves, and "commitments" with no confidence attached.

## Now / Next / Later: the honest horizon roadmap

The horizon roadmap replaces date columns with three confidence bands, and the bands *are* the honesty: confidence is high in the near horizon and deliberately low in the far one, and the format communicates exactly that.

- **Now** — committed and in-flight. This is the work the teams are actively delivering. It is the only horizon where dates approach commitment-grade, and even here the date comes from the forecast, not the roadmap. High certainty.
- **Next** — what comes up after Now, in rough priority order. These are shaped enough to sequence but not yet committed; the order can still change as Now completes and as new information arrives. Decreasing certainty — you are confident about *what* is in Next, less confident about exact order, and not committing to *when*.
- **Later** — directional bets. These name outcomes the program intends to pursue but has not shaped, sized, or sequenced. Low certainty by design. Later exists to communicate direction and let stakeholders see that their concern is "on the map" without implying it has a slot.

```
NOW (committed, in-flight)     NEXT (shaped, decreasing certainty)   LATER (directional bets)
─────────────────────────      ──────────────────────────────────   ────────────────────────
Checkout failure-rate          Self-serve refunds                    Loyalty programme
  reduction                    Reporting-export performance          International tax handling
Account-recovery flow          Fraud-signal enrichment               Partner API surface
   ▲ high confidence              ▲ medium confidence                   ▲ low confidence / no order
```

Why this beats a date-pinned roadmap: it communicates sequence and intent without manufacturing false precision. A stakeholder reading it learns the *order* and the *confidence* truthfully — they can see that "Later" means "we hear you, it is not soon, and we are not lying to you about a date we cannot defend." The format makes under-confidence in the far horizon a feature rather than an embarrassment, which is exactly the property a date Gantt lacks.

## Theme-based roadmapping: organize around bets, not feature lists

Populate the horizons with **themes** — outcome-oriented bets — rather than feature lines. A theme names a benefit and groups whatever work serves it: "reduce checkout abandonment" is a theme; "add Apple Pay, redesign the error states, and pre-fill addresses" are candidate features *inside* it. This matters for three reasons. First, a theme ties directly to a benefit (`benefits-realization-and-outcomes.md`), so the roadmap stays connected to value rather than drifting into a feature inventory. Second, themes give teams latitude to discover the best features for the outcome instead of committing to a feature list decided before the work began. Third, themes are the right granularity for portfolio sequencing — you sequence *bets against each other*, and a bet is comprehensible to governance in a way that a hundred feature lines are not. A feature-list roadmap quietly redefines success as "shipped the list"; a theme roadmap keeps success defined as "moved the metric."

## Cost of Delay: the economics underneath sequencing

**Cost of Delay (CoD)** is the cost incurred per unit time that a piece of value is *not yet delivered*. It is the single most clarifying idea in prioritization because it reframes the question from "how valuable is this?" to "what does it cost us every week we don't have it?" — and that second question is the one that actually drives sequence. A feature worth a large sum but whose value is indifferent to timing has a low cost of delay; a smaller feature tied to a deadline, a competitor move, or a compounding loss can have a high one.

CoD is commonly decomposed into three components, each scored on a relative scale (not in currency — these are *relative* magnitudes, which keeps the conversation about ranking rather than spurious dollar precision):

- **User-Business Value (UBV)** — how valuable the outcome is to users and the business once delivered.
- **Time Criticality (TC)** — how much the value decays with delay: is there a deadline, a market window, a perishability? Flat value over time means low TC.
- **Risk-Reduction / Opportunity-Enablement (RR/OE)** — does doing this *now* retire a major risk or unlock other work? A piece that de-risks the program or enables several downstream themes scores high here even if its direct user value is modest.

CoD is the sum of the three. It is the numerator of the sequencing decision — and on its own, ordering by CoD answers "what is most costly to delay." But CoD ignores *how long the work takes*, and that is the gap WSJF closes.

## WSJF: the worked arithmetic

**Weighted Shortest Job First (WSJF)** sequences work by dividing cost of delay by job size:

```
WSJF = Cost of Delay / Job Size
     = (User-Business Value + Time Criticality + Risk-Reduction/Opportunity-Enablement) / Job Size
```

The logic is queueing economics: to minimize total cost of delay across a set of jobs competing for the same capacity, do the job with the **highest cost-of-delay-per-unit-duration first**. A high-CoD job that takes forever may be worth *less per week of capacity* than a moderate-CoD job you can clear in days — and clearing the small one first starts realizing its value sooner *and* frees capacity for the next. Both CoD and Job Size are scored on the same relative scale (modified Fibonacci — 1, 2, 3, 5, 8, 13, 20 — is conventional); Job Size is a *duration/effort proxy*, and when you need it grounded in real throughput rather than a relative guess, that is the seam to `estimation-and-forecasting.md`.

Worked example. Four candidate items compete for one team's capacity — a large checkout-conversion overhaul (A), a small reporting-export fix (B), a medium fraud-signal feature (C), and a tiny settings tweak (D):

| Item | UBV | TC | RR/OE | CoD (sum) | Job Size | WSJF (CoD ÷ Size) | WSJF rank | Value (CoD) rank |
|------|-----|----|-------|-----------|----------|-------------------|-----------|------------------|
| A — checkout overhaul (large) | 9 | 7 | 5 | **21** | 13 | 21 ÷ 13 = **1.62** | 4th | **1st** |
| B — reporting-export fix (small) | 6 | 5 | 3 | **14** | 3 | 14 ÷ 3 = **4.67** | **1st** | 3rd |
| C — fraud-signal feature (medium) | 8 | 4 | 4 | **16** | 8 | 16 ÷ 8 = **2.00** | 3rd | 2nd |
| D — settings tweak (tiny) | 3 | 2 | 2 | **7** | 2 | 7 ÷ 2 = **3.50** | 2nd | 4th |

Verify the arithmetic: A = 21/13 = 1.62, B = 14/3 = 4.67, C = 16/8 = 2.00, D = 7/2 = 3.50. The WSJF sequence is **B → D → C → A**.

Now the key insight, which is the entire reason WSJF exists: **the highest-WSJF item is not the highest-value item.** Item A has the highest cost of delay by a clear margin (CoD 21) — by raw value it is the most important thing on the list — yet it sequences *dead last* by WSJF because it is large (size 13). Item B, only the *third*-most-valuable item by CoD, sequences *first*, because at size 3 it delivers more cost-of-delay-reduction per unit of capacity than anything else. Even item D, the *least* valuable item on the board, outranks the flagship A — because it is nearly free to do. This is the discipline doing its job: WSJF deliberately resists the gravitational pull of "do the biggest, most important thing first," because doing the small high-CoD work first realizes more total value sooner and unblocks the queue. (It does not mean A never happens — it means A is sequenced honestly against its cost, and if A's size is the problem, the right move is often to *slice* A smaller, which raises its WSJF; see `scope-and-backlog-management.md`.)

## When WSJF is overkill

WSJF earns its complexity at portfolio and program scale, where many themes from different owners compete for shared capacity and the sequencing decision needs an explicit, auditable economic model that a steering group can reason about together. For a **single team** with one backlog, the three-component scoring is usually ceremony: a simpler **"shortest valuable job first"** — order by cost of delay, break ties toward the smaller job — captures almost all of the benefit without the scoring overhead. The principle (highest CoD-per-duration first) is invariant across scales; only the formality of computing it changes. Reach for full WSJF when the decision is contested across teams and needs a shared model to adjudicate it; skip it when one product owner can order a single backlog by feel and be right.

## Other lenses, briefly

WSJF is not the only prioritization model, and each has a fit:

- **RICE** — `(Reach × Impact × Confidence) / Effort`. Fits product prioritization where *reach* (how many users a change touches) is the dominant axis and you want an explicit confidence discount on optimistic impact estimates. Its denominator is effort, same shape as WSJF; its numerator emphasizes reach over time-criticality.
- **Kano** — classifies features as *basic* (expected; their absence angers, their presence is unremarkable), *performance* (more is linearly better), or *delight* (unexpected, disproportionately satisfying). Fits *what kind of value* a feature offers, which informs scoring above — it complements WSJF rather than competing with it.
- **MoSCoW** — Must / Should / Could / Won't. Fits scope negotiation against a fixed deadline, where the question is what survives the cut. Its failure mode is endemic: **everything becomes a Must**, at which point it has classified nothing and the deadline absorbs the lie.

These are lenses, not rivals; mature prioritization uses CoD/WSJF for *sequence* and reaches for Kano or RICE to *inform the inputs*.

## Portfolio sequencing: the roadmap as a governance output

At program scale the roadmap is not authored by one person — it is the *output* of a governance prioritization decision (`program-structure-and-governance.md`), and it must sequence themes against **three constraints simultaneously**, not value alone:

1. **Value** — WSJF / cost of delay, as above. This sets the *desired* order.
2. **Dependency** — a high-WSJF theme that depends on a lower-WSJF enabler cannot actually go first; the enabler is pulled forward regardless of its own score (`cross-project-dependencies-and-integration.md`). Sequencing by value while ignoring the dependency graph produces an order that cannot physically execute.
3. **Capacity** — the sequence must fit the *real* throughput of the teams that will do the work, not an idealized resourcing (`capacity-and-resource-flow.md`). A value-perfect order that assumes capacity the program does not have is a fiction.

The portfolio roadmap is the reconciliation of those three: value sets the ambition, dependencies constrain the order, capacity constrains the pace. Governance owns the trade-offs between them, and the resulting Now/Next/Later horizon is the *record* of that decision — which is exactly why it must not masquerade as a set of dated commitments.

## Anti-Patterns

1. **Prioritization by the loudest voice / HiPPO.** The backlog is ordered by who asked most insistently, most recently, or most senior — the "highest-paid person's opinion" — rather than by value and cost of delay. High-WSJF work that would realize value fast sits behind low-value work with a powerful sponsor, and the program's whole economic engine runs backwards while feeling responsive. *Fix: make the prioritization model explicit (CoD, then WSJF at portfolio scale) and force every "this is urgent" claim to be scored on the same scale as everything else — authority sets context for the inputs, it does not override the arithmetic.*

2. **A date-pinned feature roadmap presented as a firm commitment.** A Gantt of features-on-dates is published as a promise, fusing intent and forecast into one dishonest artifact that must be redrawn every time reality moves and breaks a "commitment" each time. *Fix: publish a Now/Next/Later horizon roadmap for intent and sequence; source every dated commitment from a forecast with a confidence interval (`estimation-and-forecasting.md`), never from a roadmap cell.*

3. **WSJF cargo-culted with invented inputs.** The team computes WSJF religiously but the CoD components are made-up numbers that do not reflect any real economics — no thought about actual time-criticality or what is genuinely unlocked — so the formula launders guesses into authoritative-looking rank. *Fix: ground each component in a real economic story (what specifically decays with delay, what specifically is de-risked or enabled) before scoring; a WSJF table is only as honest as its CoD inputs.*

4. **MoSCoW where everything is a Must.** Faced with the cut, every stakeholder relabels their item "Must," so the classification distinguishes nothing and the deadline silently absorbs the over-commitment. *Fix: cap the Must category (e.g. Musts must fit comfortably inside forecast capacity with slack) and force Should/Could to carry real items; a MoSCoW with no Coulds has failed.*

5. **Sequencing by value while ignoring dependencies and capacity.** The roadmap orders themes by pure WSJF and assumes they can run in that order — but the top theme depends on an unsequenced enabler, or the order assumes capacity the teams do not have, so the value-optimal sequence cannot physically execute. *Fix: reconcile the value order against the dependency graph (`cross-project-dependencies-and-integration.md`) and against real team throughput (`capacity-and-resource-flow.md`) before publishing; value sets ambition, dependencies and capacity set what is actually possible.*

## Cross-References

- `benefits-realization-and-outcomes.md` — the value/benefit a theme delivers is the numerator of every prioritization decision here; CoD's User-Business Value component is only meaningful if it traces to a real benefit owned by someone.
- `estimation-and-forecasting.md` — Job Size (the WSJF denominator) is a duration/effort proxy that grounds in real throughput here; and the load-bearing rule that *dated commitments come from forecasts, not from the roadmap* hands off to this sheet.
- `program-structure-and-governance.md` — at portfolio scale the prioritization decision is owned by governance, and the roadmap is the recorded output of that decision; this sheet provides the economic model governance uses.
- `capacity-and-resource-flow.md` — the value-optimal sequence must be paced against real team capacity, not idealized resourcing; sequence against the throughput you actually have.
- `scope-and-backlog-management.md` — slicing a large, low-WSJF item smaller raises its WSJF and lets high-value work flow sooner; prioritization and slicing are the same lever from two ends.
- `cross-project-dependencies-and-integration.md` — dependencies override the pure value order: an enabler is pulled forward regardless of its own score, and ignoring the dependency graph produces a sequence that cannot execute.
