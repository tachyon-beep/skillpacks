# PRD and Acceptance Criteria

**A PRD that opens with the solution has already failed the only job it has: to make a bet falsifiable before anyone builds it. An acceptance criterion that cannot fail is not a criterion — it is a mood, and a bet whose success criterion cannot be falsified is a bet you can never lose, which means it is a bet you can never learn from.** This sheet writes the spec that turns a decided bet into something `/axiom-planning` can plan and `ACCEPT` can later judge — problem first, never solution first; non-goals named; and every criterion stated as a condition that some future observation could prove wrong. The PRD is the contract between *what product decided* and *what gets built*; if it smuggles in the how, or if its "success" cannot be falsified, the seam leaks and the build trap walks in.

The PRD is not a sixth workspace artifact. The five standing artifacts are defined in `product-state-and-continuity.md`; the PRD is the **handoff spec**, referenced from `current-state.md` by ID (`awaiting acceptance against PRD-0006`) and produced by `/write-prd`. It pairs with — and does not replace — the Product Decision Record: the **PDR records the decision** to make the bet (context → options → call → rationale → reversal trigger); the **PRD specifies** that bet (problem → criteria → handoff). They link by ID; they are different objects with different jobs.

## Problem first, solution never (here)

Lead with the problem, the user, and the desired outcome — in that order — and stop before the solution. The discipline is not stylistic; it is what keeps the PRD upstream of `/axiom-solution-architect`. The moment the spec says "build a caching layer," it has chosen a *how* product does not own, foreclosed alternatives the architect should weigh, and coupled the bet's success to one implementation. State the problem so well that several solutions are visibly possible, then hand the choice down.

A problem statement carries four parts, and a missing part is a hole a solution-first spec will fill with an assumption:

- **Who** — the specific user or segment whose problem this is. Not "users"; the segment whose job is blocked. (Whether this is the *right* problem for the right who is the discovery decision — `product-discovery-and-opportunity.md`; the PRD assumes that call was made and a PDR recorded it.)
- **The problem** — the job they are trying to do and where it currently breaks, stated as their pain, not as the absence of your feature. "Users can't export reports" is the absence of a feature; "analysts rebuild the same report by hand every Monday because last week's export is stale" is a problem.
- **The desired outcome** — the changed behavior or state that means the problem is solved, in the user's terms. This is what the success metric will measure; if you cannot name the outcome, you cannot write a falsifiable criterion for it.
- **Why now** — what makes this worth a slot. (Cost-of-delay *arithmetic* and sequencing belong to `/axiom-program-management`'s `roadmapping-and-prioritization.md`; the PRD states the *why-now story* the bet rests on, not the WSJF score.)

The tell for a solution-first spec: delete the proposed solution and see whether the problem still reads as a problem. If it collapses into "we haven't built X yet," there was no problem — there was a solution looking for one, and the is-this-worth-solving decision was skipped (`product-anti-patterns.md`).

A worked contrast, same bet, both versions one sentence:

- **Solution-first (rejected):** "Add a one-click reorder button to the cart." Delete "one-click reorder button" and nothing remains — no user, no pain, no outcome. The button *is* the spec, and it has already chosen the how.
- **Problem-first (accepted):** "Returning buyers (who) abandon when re-purchasing a prior order because they must re-find each item and re-enter quantities (the problem / their pain); we want repeat-purchase completion to rise (desired outcome); churn data shows this cohort is leaving for a competitor with faster reorder (why now)." Several solutions fit — saved carts, reorder, subscriptions — and the architect chooses among them. The success metric writes itself from the outcome.

## PRD structure

A PRD is as long as it must be and no longer; the structure below is the load-bearing minimum. It is a spec, not an essay, and not a plan.

```markdown
# PRD-0006 — <bet name>            Status: ready-for-planning
Decision: PDR-0005   Bet (roadmap.md): Now   Target metric (metrics.md): <name>

## Problem
Who · the problem (their pain) · desired outcome · why now.   (3–6 sentences.)

## Success metric (the signal the bet paid off)
The ONE metric from metrics.md this bet moves, its BASELINE, its TARGET, and
the date by which the move must show. This is the falsification condition.

## Acceptance criteria (falsifiable)
Numbered, binary, each with a pre-committed threshold and a reject branch.

## Non-goals (this bet)
Explicitly out of scope for THIS bet — the scope fence, not the product's anti-goals.

## Constraints & guardrails
Hard limits the solution must respect (a guardrail from metrics.md must not degrade;
a compliance/latency/data bound). NOT a design — a boundary the design lives inside.

## Open questions / assumptions
What is unresolved and what the bet assumes; each is a risk the plan inherits.

## Handoff
What goes to /axiom-planning (the top item), what goes to /axiom-solution-architect
(solution shape), tracker IDs.
```

What is conspicuously absent is as important as what is present: **no implementation steps, no file list, no architecture, no task breakdown.** Those are the plan (`/axiom-planning`) and the design (`/axiom-solution-architect`). A PRD that contains them has done two siblings' jobs badly and blurred the seam that makes ownership coherent.

## Falsifiable acceptance criteria — the load-bearing idea

**A criterion is falsifiable when you can state, in advance, the observation that would make you declare the bet failed.** A metric attached is not enough — "users find checkout easier" can carry a survey number and still be unfalsifiable, because no specific observation is pre-committed to count as failure. The operational test has three parts, and a criterion missing any one is not a criterion:

1. **Binary against a pre-committed threshold** — it resolves to pass or fail, not to a feeling or a direction. "Faster" is a direction; "p95 ≤ TARGET" is a threshold.
2. **Bounded by a date** — the observation has a window. "Eventually" is not falsifiable, because there is no moment at which you are forced to call it.
3. **Has an explicit reject branch** — it names what happens when the threshold is missed. The reject branch is what makes it a *test* and not a *hope*; without it, a missed target quietly becomes "we'll keep going," which is the build trap wearing a metric.

The side-by-side is the whole lesson:

| | Unfalsifiable | Falsifiable |
|---|---|---|
| **Criterion** | "Users find the checkout flow easier." | "Checkout completion rate rises BASELINE → TARGET within N days of release." |
| **Why it fails / works** | No observation is named that could prove it false — any outcome can be narrated as "easier." | A specific reading at a specific time decides it; the bet can lose. |
| **Reject branch** | none — a miss is reinterpreted, never called | "If completion stays below TARGET at N days, the bet is **rejected**; open a follow-up PDR." |
| **What ACCEPT does with it** | nothing it can defend — "shipped, looks easier" | reads the metric, renders pass/fail, records the verdict |

The deep point: **the acceptance criterion is the falsification condition at the *spec* level, exactly as the PDR's reversal-trigger is at the *decision* level** (`product-state-and-continuity.md`). Same discipline, two altitudes — the criterion tests "did this delivered thing meet its bar," the reversal-trigger tests "should we still be making this bet at all." Write the criterion unfalsifiable and you have built an acceptance gap into the spec before a line of code exists; `ACCEPT` will later have nothing it can defend a rejection on, so it will rubber-stamp "it shipped" (`product-ownership-operating-model.md`).

A worked criteria block shows the three parts and the reject branch carrying their weight — note that one criterion is the headline success metric, one is a guardrail that must not degrade, and each names what a miss means:

```markdown
## Acceptance criteria (falsifiable)
1. SUCCESS — Checkout completion rate rises BASELINE → TARGET within N days of
   release, measured on the metrics.md north-star.
   Reject branch: below TARGET at N days → bet rejected; open follow-up PDR.
2. GUARDRAIL — Payment fraud rate stays ≤ ceiling over the same window.
   Reject branch: above ceiling → bet rejected even if (1) passes (a win that
   hides a harm is not a win).
3. SCOPE — The new flow is reachable by 100% of the segment in the Problem
   (not a cohort-gated experiment) by release.
   Reject branch: gated to a subset → not yet acceptable; this criterion is unmet.
```

Contrast the failure mode the block defends against: "Checkout is smoother and conversion trends up." Smoother is unobservable-as-failure; *trends up* names no threshold, no window, and no miss condition, so no reading can ever reject the bet — which is precisely why it always "passes."

Guardrail criteria matter as much as the headline. A bet can hit its success metric *by* degrading something else — completion rises because the flow now auto-accepts risky payments. Pull the relevant guardrail from `metrics.md` into the PRD as a criterion that must **not** move the wrong way (`fraud rate ≤ ceiling at N days`), so a win that hides a harm fails acceptance.

## The success metric: name it, don't design it

Each PRD carries exactly one headline **success metric** — the signal that the bet paid off — and that metric is the spine of the acceptance criterion above. The rule that keeps this from sprawling into a sibling's territory: the PRD **names which existing metric moves and by how much**; it does **not** design a metrics discipline. Which metric is a north-star versus an input versus a guardrail, how it is instrumented, how an A/B test is powered, and when a bet is killed all live in `product-metrics-and-experimentation.md`; the durable scoreboard with BASELINE and current readings lives in `metrics.md`. The PRD points at a metric that already exists on that scoreboard (or flags that one must be added before the bet can be accepted) and states its TARGET and date. One metric, not a dashboard: a PRD that lists six "success metrics" has chosen none, because there is no single observation that decides the bet.

## The seam: where the PRD ends and the how begins

This is the differentiation, and the boundary table *is* the seam-enforcement — it draws the stop line where product's spec ends and engineering's how begins:

| Belongs in the **PRD** (this sheet) | Belongs in the **plan** (`/axiom-planning`) | Belongs in the **architecture** (`/axiom-solution-architect`) |
|---|---|---|
| Who, the problem, desired outcome, why now | Ordered tasks with exact files and code | Solution shape and component design |
| The success metric + TARGET + date | Codebase-validated sequencing of the work | Technology and pattern choices, trade-offs |
| Falsifiable acceptance criteria | Test plan that exercises the criteria | The ADR recording the design decision |
| Non-goals and guardrail constraints | Effort/risk against codebase reality | How constraints are met technically |
| Open questions and assumptions | Concrete steps that resolve them | Architectural answers to the open questions |

The flow: product writes the PRD → hands the **top item** to `/axiom-planning`, which turns it into an executable, codebase-validated implementation plan → routes any solution/architecture shaping to `/axiom-solution-architect`. The committed bet is *also* handed to `/axiom-program-management` for sequencing and forecast (the dated commitment never comes from the PRD). **Do not write the implementation plan here, and do not choose the architecture here** — restating either is the same defect as restating WSJF: it duplicates a sibling, drifts out of sync with the real plan, and dissolves the seam. The PRD's `Handoff` section records the linkage (which item, which tracker ID, which sibling owns what) — never a copy of the plan.

This is the `DISPATCH` step of the operating loop made concrete (`product-ownership-operating-model.md`): a PRD reaches `Status: ready-for-planning` only when its problem is stated, its one success metric is named, and its acceptance criteria are falsifiable — those are the gates that make it safe to hand outward. A PRD with a directional criterion is *not* ready, no matter how complete it looks, because the thing it hands to planning cannot later be accepted. The status is a claim about falsifiability, not about word count.

## Non-goals: the scope fence, not the product's refusals

Named non-goals are what keep a PRD from metastasizing in planning. A non-goal is a thing a reasonable reader might assume is in scope for *this bet* and is explicitly not — "this bet does not touch the mobile flow," "no migration of historical records," "single locale only." Naming them does two things: it prevents the planner and the architect from gold-plating around an assumed requirement, and it makes the eventual acceptance honest by stating up front what success does *not* require.

Distinguish two refusals that are easy to blur. **`vision.md` anti-goals** are *product-level* — what the product refuses to *become*, ever, under pressure (the adjacent product you decline to build). **PRD non-goals** are *bet-level* — the scope fence around *this* spec, which a later bet might well cross. Putting a product anti-goal in a PRD's non-goals understates a permanent refusal as a temporary one; putting a bet's scope fence in `vision.md` calcifies a this-time decision into product doctrine. Keep them at their own altitude (`vision-strategy-and-roadmap.md` owns the anti-goals).

## Anti-Patterns

1. **Solution-first specification.** The PRD opens with the feature ("add a caching layer," "build a dashboard") and the problem is reverse-engineered to justify it. Seductive because the solution is the exciting part and writing it feels like progress, while problem-framing feels like throat-clearing. But it forecloses the architect's alternatives, couples success to one implementation, and skips whether the problem was worth solving at all. *Fix: lead with who → problem → desired outcome → why now, and confirm the problem still reads as a problem with the solution deleted; route the how to `/axiom-solution-architect` — see the problem-first section above, and `product-discovery-and-opportunity.md` for the is-this-worth-solving call.*

2. **Unfalsifiable acceptance.** Criteria read "intuitive," "improved," "users are happier" — directional words no observation can disprove. Seductive because a criterion that cannot fail can never embarrass you at acceptance. But it builds an acceptance gap into the spec before any code exists: `ACCEPT` has nothing to defend a rejection on, so it banks "it shipped" as success. This is the *spec-level twin* of the `metrics.md` unfalsifiable-target trap. *Fix: every criterion binary, against a pre-committed threshold, bounded by a date, with an explicit reject branch; the acceptance criterion is the falsification condition at the spec level, mirroring the PDR reversal-trigger (product-state-and-continuity.md).*

3. **Gold-plating — the spec elaborates the solution.** The PRD keeps adding capability "while we're in there" — extra states, edge cases, configuration nobody asked for — because more feels more complete and saying "not this bet" feels like under-delivering. But it inflates scope past the problem, and the inflation is *how-elaboration* the product does not own. *Fix: name the non-goals explicitly to fence the bet's scope (bounds the WHAT), and route solution-elaboration to `/axiom-solution-architect` (owns the HOW); the PRD states the problem and the bar, not the trimmings.*

4. **One metric short, or six metrics long.** The PRD names no success metric (so nothing decides the bet) or lists many (so nothing single decides it). Seductive because more metrics feel more rigorous and no metric avoids committing. Either way there is no single observation that pays off or kills the bet. *Fix: name exactly one headline success metric with BASELINE → TARGET and a date, drawn from the metrics.md scoreboard; route metric design and kill-logic to product-metrics-and-experimentation.md.*

5. **PRD as plan.** The spec lists files to touch, steps to take, an architecture to adopt. Seductive because it feels thorough and like a head start for engineering. But it does `/axiom-planning`'s and `/axiom-solution-architect`'s jobs badly, drifts out of sync with the real plan the moment the codebase pushes back, and erases the seam. *Fix: keep the PRD to problem → criteria → non-goals → handoff; hand the top item to `/axiom-planning` and the solution shape to `/axiom-solution-architect`, recording only the linkage.*

## Cross-References

- `product-ownership-operating-model.md` — `DECIDE` writes the bet and its falsifiable criteria; `ACCEPT` later tests the delivered thing against this PRD's criteria. A weak criterion here is an acceptance gap there; `DISPATCH` is where this PRD's top item goes to `/axiom-planning`.
- `product-state-and-continuity.md` — defines the five workspace artifacts (the PRD is *not* one of them — it is the handoff spec referenced by ID from `current-state.md`), the PDR template the PRD links to, and the `metrics.md` scoreboard this PRD's success metric is drawn from. The PDR reversal-trigger is the decision-level twin of the acceptance criterion.
- `product-discovery-and-opportunity.md` — owns the is-this-worth-solving / for-whom decision the PRD assumes was made; the problem statement here restates that call, it does not re-derive it.
- `vision-strategy-and-roadmap.md` — owns product-level anti-goals (what the product refuses to become), distinct from this sheet's bet-level non-goals (the scope fence on one PRD).
- `product-metrics-and-experimentation.md` — owns metric design, instrumentation, A/B/hypothesis mechanics, and kill-the-bet logic; the PRD names which metric moves and by how much, it does not design the discipline.
- `/axiom-planning` — turns the PRD's top item into an executable, codebase-validated implementation plan. The PRD owns what/why and the bar; planning owns the plan. Do not write the plan here.
- `/axiom-solution-architect` — owns the solution shape, architecture, and ADRs (the *how*). Route solution-elaboration there; the PRD names constraints the design lives inside, not the design.
- `/axiom-program-management` — sequences and delivers the committed bet: WSJF/cost-of-delay (`roadmapping-and-prioritization.md`), forecast (`estimation-and-forecasting.md`), flow. The dated commitment comes from its forecast, never from the PRD. The PRD's why-now is a story, not a WSJF score.
