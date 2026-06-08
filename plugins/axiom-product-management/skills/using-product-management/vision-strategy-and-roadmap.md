# Vision, Strategy, and Roadmap

**Vision, strategy, and roadmap are three different objects with three different jobs, and the recurring failure is to collapse them into one — a vision that is a motivational slogan, a strategy that is a feature list, and a roadmap that is a dated promise. Each counterfeit feels like clarity and provides none.** Vision says what the product is *for* and what it refuses to be; strategy says where you will play and how you will win, given finite capacity; the roadmap says which outcome-bets you are pursuing, in what rough order, with confidence that decreases as you look further out. This sheet shapes all three — but the moment a bet is committed, the *sequencing* of it (Now/Next/Later mechanics, WSJF, cost of delay) is owned by `/axiom-program-management` and this sheet hands it over rather than restating it. Product decides *which bets and why*; program decides *in what order they flow and when they land*.

These artifacts are persisted, not invented fresh each session: vision lives in `vision.md`, the bets in `roadmap.md`, each schema defined in `product-state-and-continuity.md`. This sheet is the *discipline* behind those files — what makes a vision load-bearing instead of decorative, a strategy a real choice instead of a wish-list, a roadmap honest instead of a broken-promise generator.

## Vision: purpose, who-it-serves, and the anti-goal

A vision is not an aspiration ("be the best X") and not a feature horizon ("a platform that does everything"). It is a *constraint*: a statement specific enough that it tells you what **not** to build. The test of a real vision is that it can be violated — if every conceivable feature is consistent with the vision, the vision decides nothing and is decoration.

Three components, each falsifiable in its own way:

- **Purpose** — the change in the world the product exists to make, stated as an outcome for someone, not a description of the software. "Let small teams ship audited releases without a release engineer" is a purpose; "a CI/CD platform" is a category.
- **Who it serves** — the primary user/segment whose problem is the *reason to exist*, the secondary served-but-not-at-the-primary's-expense, and the explicit **not-for**. The not-for is the load-bearing line: a vision that serves "everyone" serves no one and cannot adjudicate a trade-off between two users who want opposite things.
- **Anti-goals** — the tempting adjacent products you will deliberately *refuse to become*, and the capabilities you will decline even under pressure. Anti-goals are how a vision survives contact with a loud customer request: "we do not become a general-purpose workflow engine" is a sentence you can hold up against a feature ask and let it lose.

The anti-goal is what separates a vision from a slogan. A slogan inspires and excludes nothing; a vision excludes, and the exclusion is where its value lives. When `DECIDE` faces a request that is individually reasonable but pulls toward an anti-goal, the vision is the recorded authority that lets product say no without re-litigating it every session — and because it lives in `vision.md`, the refusal is inspectable, not a matter of the agent's mood. (Changing vision is *not* an autonomous act — it escalates; see the authority boundary in `product-ownership-operating-model.md`.)

## Positioning: the frame the user judges you in

Positioning is the answer to "for whom, against what alternative, and why us" — the competitive and contextual frame in which the user decides whether the product is worth their attention. It is downstream of vision and upstream of strategy: vision says what you are for; positioning says what you are *instead of*. A useful positioning statement names four things explicitly:

> For **\<primary user\>** who **\<has this problem/job\>**, the product is a **\<category\>** that **\<the key differentiating benefit\>**, unlike **\<the dominant alternative — including "do nothing in a spreadsheet"\>**, because **\<the reason the difference is credible\>**.

The discipline is naming the *real* alternative, which is usually not a named competitor — it is the status quo, the manual workaround, the incumbent tool the user already tolerates. A positioning that pretends the alternative is a weaker competitor while the actual alternative is "doing nothing" will lose to inertia it never accounted for. Positioning constrains strategy: it tells you which axis you are competing on (cost, speed, trust, breadth, integration) and therefore which investments are on-strategy and which are vanity. The *research method* that validates whether a positioning lands with real users — interviews, usability testing — belongs to `/lyra-ux-designer`; this sheet owns the product/opportunity lens of *which* position to take and whether the problem behind it is worth solving (`product-discovery-and-opportunity.md`).

## Strategy: where to play, how to win

Strategy is a set of *choices under finite capacity*, not a list of things you would like to be true. The two questions, in order:

- **Where to play** — which users, which problems, which segment of the market you will compete for, and (equally) which you will not. A where-to-play that includes everything is the absence of a strategy.
- **How to win** — the specific advantage that makes you the right choice *in that arena*: a structural edge (data, distribution, integration), a focus the incumbent cannot match without cannibalizing itself, or a capability that compounds. "Be better" is not a how-to-win; "win on time-to-first-audited-release because we own the toolchain end-to-end" is.

The test that a strategy is real: it implies sacrifice. If the strategy does not name something valuable you are *choosing not to do* in order to win where you have chosen to play, it is a wish list wearing a strategy's clothes. A strategy-as-feature-list — "our strategy is SSO, mobile, and an API" — has made no choice; it has enumerated outputs and called the enumeration a direction. The corrective is to force each candidate investment to answer: *which where-to-play does this serve, and which how-to-win does it advance?* Features that cannot answer are off-strategy, however appealing, and the strategy's job is to let them lose cleanly.

A traceability table makes the discipline mechanical — every candidate names the strategic choice it serves, or it is off-strategy and named as such:

| Candidate investment | Where-to-play served | How-to-win advanced | Verdict |
|---|---|---|---|
| Pre-fill audit metadata from the toolchain | primary segment | own-the-toolchain-end-to-end | on-strategy |
| Reduce false-positive flags | primary segment | win-on-time-to-first-audited-release | on-strategy |
| General-purpose workflow builder | *none — adjacent market* | *none — pulls toward an anti-goal* | **off-strategy: decline** |
| Mobile app | unproven for primary | none stated | **off-strategy until a where-to-play justifies it** |

The two off-strategy rows are the table doing its job: each is individually appealing and neither survives the question "which choice does this serve." That is the strategy adjudicating, which a feature list cannot do.

Strategy is the bridge between vision (the destination) and the roadmap (the bets that move toward it). Every bet on the roadmap should trace to a strategic choice; a bet that traces to nothing is the product drifting, and that drift is exactly what a written strategy exists to catch in `DECIDE`.

## The north-star as a strategic anchor

The north-star metric is the *single measure that proxies for delivered value* under the current strategy — the one number that, if it moves the right way, means the strategy is working. Its job here is **framing**, not measurement: it forces the strategy to commit to what "winning" *means* before any bet is placed, so that bets can be judged against it rather than against "did it ship."

A north-star earns the name only if it satisfies three properties:

- **It proxies value to the user, not activity by the product.** "Weekly teams shipping an audited release" tracks the purpose; "total releases processed" tracks the machine running. The first moves only if users get value; the second moves if the product is merely busy.
- **It is leading, not purely lagging.** Revenue is the ultimate lagging measure and a poor north-star because it moves too late to steer by. The north-star should be the user-value proxy that *predicts* the lagging business result, so a bet can be judged within a horizon you can act on.
- **It comes paired with a guardrail.** Any single metric can be gamed; the guardrail names the harm a "win" could hide (latency, support load, churn) so that moving the north-star at the guardrail's expense is correctly read as a loss, not a win.

This sheet defines only the *framing* — what makes a candidate a legitimate north-star and how it anchors strategy. The **measurement** — instrumentation, baselines, falsifiable targets with dates, input-metric decomposition, A/B and hypothesis design, and the kill-the-bet logic — is owned by `product-metrics-and-experimentation.md`, and the durable scoreboard (target, current reading, trend) lives in `metrics.md` per `product-state-and-continuity.md`. Choose the north-star here; make it falsifiable and measure it there.

## Strategic bets: the unit of the roadmap

A bet is a *hypothesis* — "if we do X for user Y, north-star metric Z moves, because of reasoning W" — not a feature commitment. Framing roadmap items as bets rather than features is what keeps the product falsifiable: a feature can only "ship or not ship," but a bet can be *right or wrong*, and being able to be wrong is the whole point of `ACCEPT`. Each bet on the roadmap should carry, even informally:

- the **outcome** it pursues (the theme/benefit, not the feature),
- the **strategic choice** it serves (which where-to-play / how-to-win),
- the **north-star or input metric** it is meant to move, and
- the **reversal trigger** — the pre-committed condition under which it is wrong and should be revisited (recorded in the PDR; `product-state-and-continuity.md`).

A bet stated this way hands cleanly to the rest of the loop: the falsifiable acceptance criteria are written in the PRD (`prd-and-acceptance-criteria.md`), the bet is dispatched for delivery, and `ACCEPT` later tests whether the hypothesis held. A "bet" with no metric and no reversal trigger is just a feature with optimistic framing — it cannot be killed when it fails, which is how the build trap takes hold.

The central claim of this sheet is that the three objects *chain* — and a single cascade shows it end-to-end:

```
VISION (purpose)     Let small teams ship audited releases without a release engineer.
   │
STRATEGY (choice)    Where-to-play: small teams in regulated domains, not the enterprise.
   │                 How-to-win:    own the toolchain end-to-end so audit metadata is free.
   │
BET (hypothesis)     "Cut time-to-first-audited-release" — IF we pre-fill audit metadata
   │                 from the toolchain for that segment, THEN the north-star moves,
   │                 BECAUSE the manual metadata step is the dominant first-release delay.
   │
METRIC (anchor)      North-star: weekly teams shipping an audited release.
   │                 Input metric it moves first: median time-to-first-audited-release.
   │
REVERSAL TRIGGER     If time-to-first-audited-release does not fall by TARGET within
                     two readings, the bet is wrong — reopen it (PDR).
```

Each link constrains the next: the bet is on-strategy because it serves a named where-to-play and how-to-win; it is falsifiable because it names the metric it moves and the condition under which it is wrong. Break any link — a bet with no strategic trace, or no metric, or no reversal trigger — and the chain stops being a strategy and becomes a list.

## The roadmap is intent, not a schedule

The roadmap is a statement of **intent about outcomes over time** — which bets you are pursuing and roughly in what order. Product *populates* the Now/Next/Later horizon with date-free, outcome-themed bets:

```
NOW (committed, in-flight)          NEXT (shaped)                          LATER (directional)
──────────────────────────          ───────────────────────────────────   ────────────────────────
Cut time-to-first-audited-release   Self-serve role management            Multi-region audit storage
Reduce false-positive audit flags   Audit-export performance              Partner integration surface
```

The Now/Next/Later band *structure*, its confidence-gradient semantics (what Now vs Next vs Later mean and why far-horizon under-confidence is honest), and all sequencing are `/axiom-program-management`'s (`roadmapping-and-prioritization.md`) — product populates the bands; it does not own the format or re-explain what the bands mean. What product *does* own is the property the bands cannot supply on their own: **every entry is a bet that traces to a strategic choice and names the metric it moves.** A program-management roadmap will faithfully sequence whatever it is handed; the *product* roadmap is responsible for the entries *deserving* their place — each one on-strategy, falsifiable, and revisitable. Strip a band entry of its strategic trace and its metric and it is just a feature waiting to be sequenced, not a product bet — and the roadmap has quietly reverted to a feature list with horizons drawn around it.

The cardinal rule, and the load-bearing seam of this sheet: **the roadmap never carries a date, and the moment a bet is committed, its sequencing is handed to `/axiom-program-management`.** That pack — specifically its `roadmapping-and-prioritization.md` — owns the Now/Next/Later *delivery mechanics*, the WSJF (Weighted Shortest Job First) and cost-of-delay arithmetic, the RICE / Kano / MoSCoW lenses, and the reconciliation of value against dependencies and capacity. A *dated commitment* comes only from a forecast built on historical throughput, which is also program-management's (`estimation-and-forecasting.md`) — never from a roadmap cell. Do not compute WSJF here. Do not draw a Gantt here. Product decides *which* bets earn a place and *why* (vision, strategy, north-star); program decides the *order they flow and when they land*. Restating the sequencing arithmetic in this pack is not thoroughness — it is duplicating a sibling, the workspace and the real backlog then diverge, and the seam that makes the two packs coherent dissolves.

The handoff in practice: `DECIDE` commits a bet and records the PDR; the bet moves to **Now** in `roadmap.md` as intent; `DISPATCH` hands it to `/axiom-program-management` for sequencing and forecast and to `/axiom-planning` for the implementation plan of the top item. The roadmap cell stays a statement of intent the whole time; the date and the WSJF score live downstream, owned by the pack that owns them.

## Anti-Patterns

1. **Roadmap-as-promise.** A Now/Next/Later of *intent* is read or published as a set of dated commitments — or worse, dates and WSJF scores creep into `roadmap.md` because a stakeholder wanted precision. Seductive because a date looks like control and "Later, no date" feels evasive. But fusing intent with forecast produces a document that must be redrawn every time reality moves and breaks a "commitment" each time it does. *Fix: keep the roadmap intent-only and date-free; source every date from a forecast and every sequence from WSJF, both owned by `/axiom-program-management` (`roadmapping-and-prioritization.md`); see also the roadmap-drift anti-pattern in `product-state-and-continuity.md`.*

2. **Vision-as-slogan.** The vision is an inspirational line ("empower everyone to build") that excludes nothing and therefore decides nothing — every feature is consistent with it, so it cannot adjudicate a single trade-off. Seductive because it sounds ambitious and offends no one. But a vision that cannot be violated provides no constraint, and the product drifts toward whoever asks loudest. *Fix: write a vision with a real who-it-serves and explicit anti-goals — capabilities and adjacent products you refuse, stated specifically enough to hold up against a feature request and let it lose.*

3. **Strategy-as-feature-list.** "Our strategy is SSO, a mobile app, and an open API" — an enumeration of outputs presented as a direction, with no where-to-play, no how-to-win, and no sacrifice named. Seductive because a list of impressive features feels concrete and a strategy of *choices* feels abstract and risky. But a strategy that names nothing it is choosing *not* to do has made no choice, so every feature is "on-strategy" and the strategy adjudicates nothing. *Fix: state where-to-play and how-to-win as choices that imply sacrifice; force each candidate investment to name which strategic choice it serves, and let the ones that cannot answer lose.*

4. **North-star that measures the machine, not the value.** The north-star is an activity total — releases processed, API calls, registered accounts — that rises whenever the product is busy regardless of whether any user got value. Seductive because activity metrics only go up and feel like momentum. But a metric that moves on activity rather than delivered value lets a feature factory report success while the business case never realizes. *Fix: choose a north-star that proxies user value and leads the lagging business result, paired with a guardrail; make it falsifiable and measure it per `product-metrics-and-experimentation.md`.*

5. **Bets-as-features (the unfalsifiable roadmap item).** Roadmap items are framed as features to ship rather than hypotheses that can be wrong — no metric, no reversal trigger, so "done" means "shipped" and the item can never be killed. Seductive because a feature is concrete and a hypothesis feels like hedging. But an item that cannot be wrong cannot teach you anything, and a roadmap of un-killable features is the build trap drawn as a plan. *Fix: frame each bet as "if X for user Y, metric Z moves because W," with a reversal trigger recorded in the PDR (`product-state-and-continuity.md`); the falsifiable acceptance criteria go in the PRD (`prd-and-acceptance-criteria.md`).*

## Cross-References

- `/axiom-program-management` — owns the committed bet's *sequencing and delivery*: Now/Next/Later mechanics, WSJF / cost-of-delay / RICE / Kano / MoSCoW arithmetic, and dated forecasting (`roadmapping-and-prioritization.md`, `estimation-and-forecasting.md`). This sheet decides *which* bets and *why*; that pack decides the *order and the when*. Never restate its arithmetic here — the load-bearing seam of the whole pack.
- `product-discovery-and-opportunity.md` — validates that the problem behind a position or bet is worth solving and for whom; vision and strategy are downstream of the opportunity decision this sheet owns the lens for.
- `product-metrics-and-experimentation.md` — owns north-star *measurement*: instrumentation, falsifiable targets, input-metric decomposition, A/B/hypothesis design, and kill-the-bet logic. This sheet frames *which* north-star anchors the strategy; that sheet makes it measurable.
- `product-state-and-continuity.md` — defines the file schemas for `vision.md` (purpose, anti-goals, authority grant), `roadmap.md` (intent-only, sequencing routed out), and the PDR that records each bet's reversal trigger. This sheet is the discipline behind those files.
- `product-ownership-operating-model.md` — `DECIDE` commits the bets this sheet shapes and `DISPATCH` hands them out; changing vision or strategy is *not* an autonomous act and escalates per the authority boundary defined there.
- `prd-and-acceptance-criteria.md` — turns a committed bet into a spec with the falsifiable acceptance criteria that `ACCEPT` tests the hypothesis against; the bet's framing here becomes the criteria there.
- `/lyra-ux-designer` — user-research method (interviews, usability testing) that validates whether a positioning or problem lands with real users. This pack owns *which* position to take; route the research craft there.
