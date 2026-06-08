# Product Discovery and Opportunity

**Discovery exists to earn the right to build something — and its most valuable output is the word "no."** A bet you commit to without first proving the problem is real, acute, for a knowable someone, and worth more than its cost is not a bet; it is a guess dressed as a decision. This sheet owns the *product/opportunity lens*: deciding whether a problem is worth solving, for whom, and why, and recording that decision — including "no" and "not now" as first-class outcomes — as a Product Decision Record before any PRD is written. It does **not** own how to run a user interview or a usability test; that craft is `/lyra-ux-designer`'s `user-research-and-validation.md`. Discovery here is the *decision layer that consumes research*, not the research itself.

Where this sits in the loop: discovery is the evidence-gathering that feeds `DECIDE` (`product-ownership-operating-model.md`). A discovery pass ends in one of three places — *commit* (a bet enters `roadmap.md`, recorded as a PDR), *not now* (parked with a reversal trigger), or *no* (declined, recorded so it is not relitigated). Sequencing a committed opportunity against everything else is not discovery's job; that is handed to `/axiom-program-management`.

## The opportunity, stated as a problem — never as a solution

A well-formed opportunity is a statement about a *problem and a population*, with no solution baked in. The discipline is to write it so that many solutions could satisfy it — because the moment the opportunity names a solution, discovery has already been skipped and you are validating a feature, not a problem. Use a fixed shape so the gaps are visible:

| Slot | What it asserts | Failure if vague |
|---|---|---|
| **Who** | The specific population that has the problem (a segment, not "users") | "Everyone" means no one; you cannot validate or size it |
| **Struggle** | The job they are trying to get done and where it breaks today | A solution in disguise ("they need feature X") |
| **Acuity** | How painful / frequent / costly the struggle is right now | A nice-to-have masquerading as a need |
| **Today's workaround** | What they do instead (a competitor, a spreadsheet, nothing) | No workaround often means no real pain |
| **Why us / why now** | What makes this ours to solve and timely | A real problem that is not *our* problem to solve |

If you cannot fill *Today's workaround*, treat it as a red flag, not a blank: a problem acute enough to solve almost always has an ugly current substitute, and "they do nothing today" frequently means the pain is below the threshold that drives behavior. The opportunity statement is the artifact you validate against — keep it solution-free until the commit decision is made.

## Jobs-To-Be-Done: the lens that resists solution-first thinking

Jobs-To-Be-Done (JTBD) frames the opportunity around the *progress a person is trying to make* rather than the demographic who wants a feature. The canonical shape — **"When [situation], I want to [motivation], so I can [expected outcome]"** — is valuable precisely because it has no product in it. A user does not want a faster export; they want, *when preparing the monthly board pack, to assemble the numbers without re-keying them, so they can spend the evening reviewing instead of copying.* The job is stable; the solutions that serve it are interchangeable and will change over time.

JTBD earns its place here for three reasons. First, it forces the **functional, emotional, and social** dimensions of a job into view — people "hire" products for emotional jobs (feel competent, avoid embarrassment) as much as functional ones, and a roadmap that serves only the functional job often loses to one that serves the emotional one. Second, it reframes the competitive set: the alternative to your product is not the obvious rival but *whatever the person hires today to make the same progress* — which is frequently a spreadsheet, a manual process, or doing nothing. Third, it is the natural antidote to falling in love with a solution: a job is satisfiable many ways, so holding the job fixed lets you compare candidate solutions against the *progress* rather than against your attachment to one of them.

JTBD is a framing lens, not a research method. You *discover* the job through research whose technique lives in `/lyra-ux-designer`; you *use* JTBD here to keep the opportunity statement honest and the success criteria tied to progress rather than to a feature shipping.

## Problem validation: real, for whom, how acute

Validation answers three questions in order, and an opportunity that fails any one of them does not advance — regardless of how exciting the solution is.

1. **Is the problem real?** Distinguish *stated* demand from *revealed* behavior. People say they want many things they never act on; the signal that counts is what they already do under their own initiative — the workaround they built, the money they already spend on a substitute, the hack they maintain. A problem with a costly current workaround is validated by the workaround's existence; a problem that only appears when you ask "would you use a thing that…" is unvalidated and frequently illusory.
2. **For whom, specifically?** A problem real for a narrow, reachable segment beats one diffusely mild across "everyone." Name the segment concretely enough that you could find ten of them this week. If the answer drifts toward "all our users," the opportunity is under-specified, not broad — and an under-specified *who* makes the business case unsizeable.
3. **How acute — and how frequent?** Acuity and frequency together set the value ceiling. A severe pain felt once a year and a mild annoyance felt hourly are different opportunities with different solution shapes. Rank ruthlessly: most candidate problems are real but not acute enough to clear the bar, and saying so is discovery succeeding, not failing.

The research *mechanics* that produce this evidence — interview protocol, sample selection, avoiding leading questions, usability-test design — are owned by `/lyra-ux-designer` (`user-research-and-validation.md`). This sheet owns what you *do with the evidence*: the judgment that turns it into a real/for-whom/how-acute verdict, and the decision that follows.

## The business case: opportunity sizing without false precision

The business case answers "is this worth more than it costs," and its job is to be *defensibly approximate*, not falsely precise. A sizing built from invented numbers laundered through arithmetic is worse than an honest range, because it manufactures confidence the evidence does not support. Three guardrails keep it honest:

- **Size the value, not the revenue fantasy.** Estimate the value to the population — pain removed, time saved, money the workaround currently costs them — and the reachable fraction of that population. A top-down "1% of a huge market" is the classic tell of a sizing nobody believes; a bottom-up "this segment has roughly N members, each loses about T per month to the workaround" is defensible because each input is inspectable.
- **Carry the cost honestly, including opportunity cost.** The cost of a bet is not just its build effort; it is the *next-best bet you do not make* while capacity is consumed. This is why discovery feeds prioritization rather than replacing it: discovery establishes that an opportunity *clears the bar*; whether it beats other bar-clearing opportunities for capacity is a sequencing decision owned by `/axiom-program-management` (cost-of-delay, WSJF). Do not reinvent that arithmetic here.
- **Express the case as a range with its assumptions named.** "Worth roughly X–Y, assuming the segment is about N and adoption clears M%" beats a single number, because the assumptions are exactly what `ACCEPT` and the PDR's reversal trigger will later test. A business case whose assumptions are not falsifiable cannot be validated after the fact, which means the bet can never be honestly killed.

The sizing's purpose is to support the *commit / not-now / no* decision — not to forecast delivery. The moment a bet commits, the dated forecast and the sequencing are handed downstream; the business case is the input to the decision, not a delivery plan.

## Desirability, viability, feasibility: the triad a bet must clear

A committed bet must satisfy three independent tests, and the failure mode is treating any one as sufficient. They are AND-gated: passing two and assuming the third is the most common way a validated problem becomes a failed product.

| Lens | The question | Owned / informed by | Failure if assumed |
|---|---|---|---|
| **Desirability** | Do the people actually want this solved — revealed, not stated? | This sheet (problem validation) + research method via `/lyra-ux-designer` | Build trap: a thing nobody hires |
| **Viability** | Does solving it create more value than it costs us, including opportunity cost? | This sheet (business case) | A loved feature that loses money or starves a better bet |
| **Feasibility** | Can it actually be built, within constraints we can live with? | `/axiom-solution-architect` (the *how*); this sheet only asks *whether* | Committing to something the architecture cannot deliver |

Note the boundary inside feasibility: discovery asks *whether* a feasible solution plausibly exists, enough to commit; it does **not** design the solution. The solution shape, the architecture, and the ADRs are `/axiom-solution-architect`'s territory. Pulling feasibility detail into discovery is how the opportunity lens quietly turns into a design exercise and the *what/why* decision gets skipped.

## The decision: commit, not-now, or no — and "no" is a first-class outcome

Discovery's terminal act is a decision, recorded as a Product Decision Record (`product-state-and-continuity.md`). There are exactly three outcomes, and a healthy product produces *no* and *not-now* far more often than *commit* — a discovery process that only ever says yes is not discovering anything; it is rationalizing a backlog someone already decided.

- **Commit.** The triad clears: real and acute (desirability), worth more than it costs (viability), plausibly buildable (feasibility). The opportunity becomes a bet in `roadmap.md` (Now/Next/Later as *intent*), the PDR records context → options → call → rationale → reversal trigger, and the falsifiable success criterion is set now so `ACCEPT` has a contract to test (`prd-and-acceptance-criteria.md`). Sequencing against other committed bets is handed to `/axiom-program-management`.
- **Not now.** Real but not yet — the segment is too small today, a dependency is unmet, the timing is wrong. Park it in *Later* with a **reversal trigger**: the metric or condition that would reopen it ("revisit if segment crosses N" / "revisit when dependency Z ships"). A parked opportunity with no trigger is just a forgotten one.
- **No.** The problem is not real, not acute enough, not ours, or not viable. Record it as a *declined* PDR with the reason — this is what stops the same idea returning every third session and being relitigated from scratch. A written "no" with its rationale is a continuity asset; an unwritten one is a guarantee of re-debate.

The decision is the seam to the rest of the pack: *commit* hands a bet to `vision-strategy-and-roadmap.md` (does it fit the strategy and the anti-goals?) and onward to the PRD; *no* and *not-now* are recorded and the loop moves on. None of the three outcomes is a sequencing decision — discovery decides whether an opportunity *clears the bar to be a bet at all*, and program-management decides the order of those that do.

## Continuous discovery: an ongoing input to the roadmap, not a phase

Discovery is not a gate you pass through once before building; it is a standing input that keeps the roadmap connected to reality. The roadmap's *Next* and *Later* horizons are populated and re-validated by a continuous trickle of discovery evidence, so that what reaches *Now* has been pressure-tested rather than imagined a quarter ago. In the operating loop, discovery evidence surfaces in `ORIENT` (what changed about the problem since last checkpoint) and can fire a PDR's reversal trigger — a "not-now" whose condition just became true, or a committed bet whose problem evidence eroded. The practical discipline: maintain a small, standing set of opportunities under validation rather than batching all discovery before a big commit. The anti-pattern this prevents is the roadmap calcifying into a list of bets validated once and never re-examined while the world moved.

Continuous discovery is *also* the place a "not-now" gets promoted honestly: when a parked opportunity's reversal trigger fires, it re-enters discovery for a fresh real/acute/for-whom/viable verdict — not an automatic commit. Conditions changing earns a *re-decision*, not a free pass into the roadmap.

## Anti-Patterns

1. **Solution-in-search-of-problem.** A favoured solution exists first, and discovery is run backwards to find a problem that justifies it — the opportunity statement quietly names the solution, and "validation" cherry-picks evidence that fits. Seductive because the solution is often genuinely clever and someone is excited about it. But a solution validated against a reverse-engineered problem clears no real bar; it commits capacity to a guess. *Fix: write the opportunity solution-free (Who / Struggle / Acuity / Workaround / Why-us) and validate the problem before any solution is on the table — the JTBD framing above keeps the statement about progress, not features.*

2. **Falling in love with a solution.** The team becomes attached to one solution and stops comparing it against alternatives that serve the same job, so the JTBD lens collapses to a single option and the business case is built to defend it rather than to test it. Seductive because conviction feels like leadership. But attachment makes the viability test cosmetic and blinds you to a cheaper solution for the same job. *Fix: hold the job fixed and the solution variable — JTBD makes the alternatives visible; the viability lens compares them on value-per-cost, not on attachment.*

3. **Stated demand mistaken for revealed demand.** "Would you use X?" gets enthusiastic yeses and the opportunity is declared validated, but no one's current behavior shows the pain. Seductive because positive interview responses feel like a green light. But people say yes to hypotheticals they will never act on; only revealed behavior — an existing costly workaround — validates a problem. *Fix: validate against what the population already does under its own initiative, not what it says it would do; the research craft for eliciting revealed behavior is `/lyra-ux-designer`'s `user-research-and-validation.md`.*

4. **"No" treated as failure.** Every discovery pass ends in commit because declining feels like wasted effort or like saying the idea was bad. Seductive because momentum and stakeholder hope both pull toward yes. But a discovery process that cannot say no validates nothing — it rationalizes a pre-made decision, and the build trap follows. *Fix: treat "no" and "not-now" as first-class, recorded outcomes; a declined PDR with its reason is a continuity asset that stops the idea returning every session (`product-state-and-continuity.md`).*

5. **Sizing as false precision.** A business case is built from invented numbers run through tidy arithmetic to produce an authoritative-looking figure nobody can defend. Seductive because a single number reads as rigor and ends debate. But laundered guesses manufacture confidence the evidence does not support, and a case with no named assumptions can never be falsified after the fact. *Fix: size bottom-up as a range with each assumption named and inspectable; the assumptions are exactly what the PDR's reversal trigger and `ACCEPT` will later test.*

6. **Discovery confused with prioritization.** Discovery decides an opportunity is good and the team treats that as a decision to build it *next*, jumping the sequencing queue ahead of bets already validated. Seductive because a freshly validated opportunity feels urgent. But clearing the bar to *be* a bet is not the same as winning the competition for capacity. *Fix: discovery decides whether an opportunity clears the bar; the order of bar-clearing bets is a cost-of-delay/WSJF decision owned by `/axiom-program-management` — hand the committed bet over, do not self-sequence it.*

7. **Discovery as a one-time gate.** The opportunity is validated once, enters the roadmap, and is never re-examined while the segment, the workaround, or the competitive set moves underneath it. Seductive because re-validation feels like reopening settled work. But a bet validated a quarter ago and never revisited is a stale guess the roadmap now treats as fact. *Fix: run discovery continuously as an input to Next/Later; let reversal triggers reopen parked and committed bets when the evidence shifts (`product-ownership-operating-model.md`, ORIENT).*

## Cross-References

- `product-ownership-operating-model.md` — discovery is the evidence that feeds `DECIDE`; the commit/not-now/no decision is a `DECIDE` act recorded as a PDR, and continuous-discovery deltas surface in `ORIENT` and can fire reversal triggers.
- `product-state-and-continuity.md` — the PDR template (context → options → call → rationale → reversal trigger) that records every discovery outcome, including the declined "no" and the parked "not-now"; the workspace is where a decision becomes a continuity asset rather than a re-debate.
- `vision-strategy-and-roadmap.md` — a committed opportunity must fit the strategy and not collide with the anti-goals; discovery hands the bet there to be placed in Now/Next/Later as intent.
- `prd-and-acceptance-criteria.md` — a committed opportunity's falsifiable success criterion is set at commit time and carried into the PRD; weak criteria here become an acceptance gap downstream.
- `product-metrics-and-experimentation.md` — when validation needs an experiment (an MVP test rather than an opinion) and for the kill/keep logic behind a parked opportunity's reversal trigger.
- `product-anti-patterns.md` — the broad product failure catalog (build trap, feature factory) that the solution-in-search-of-problem and "no-is-failure" traps here feed into.
- `/lyra-ux-designer` — owns research *method*: `user-research-and-validation.md` covers interview technique, sample selection, leading-question avoidance, and usability-test design. This sheet owns the *decision lens* that consumes that research; route the mechanics there.
- `/axiom-program-management` — sequences opportunities that clear the discovery bar against each other (cost-of-delay, WSJF); discovery decides *whether* something is a bet, program-management decides *the order of the bets*. Do not self-sequence here.
- `/axiom-solution-architect` — owns the *how* behind feasibility (solution shape, architecture, ADRs); discovery asks only *whether* a feasible solution plausibly exists, enough to commit.
