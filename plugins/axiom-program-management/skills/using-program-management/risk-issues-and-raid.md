# Risk, Issues, and the RAID Log

**A RAID log is the instrument that turns uncertainty into managed work. Its entire value is in the verbs: a risk is *scored*, *owned*, *re-scored*, *escalated*, and either *closed* or *converted* — and a log where none of those verbs ever fire is not a risk management tool, it is a kickoff relic that exists to be shown to an auditor.** The single most common failure is not having no log; it is having a log that was filled in once and never touched again, so that every risk on it either silently expires or arrives as a surprise issue with no warning in between. This sheet is about the *operational* discipline that keeps a RAID log alive. The *formal* process — Risk Management (RSKM) and Decision Analysis (DAR) as defined process areas for regulated, high-maturity contexts — lives in `/axiom-sdlc-engineering`; the handoff is explicit and load-bearing at the end.

## RAID, defined precisely

RAID is four different things people put in one log because they share a review cadence, not because they are the same kind of object. The distinctions are constantly blurred, and the blurring is what makes logs useless. Be sharp:

- **Risk** — a *future* event that **might** happen, with a probability strictly between 0 and 1. "The third-party API may not be ready by integration." It hasn't happened. It might never happen. You manage it by changing its probability or its impact, or by deciding consciously to live with it. A risk is a bet against the future.

- **Assumption** — something you are *treating as true but have not verified*. "We assume the vendor's rate limit is sufficient for our peak load." An assumption is a **latent risk**: if it's wrong, it converts into a risk (if the consequence is still in the future) or straight into an issue (if the consequence is already on you). Untracked assumptions are where surprises are manufactured.

- **Issue** — something that **has happened or is true right now**: probability = 1. "The vendor confirmed the API will not be ready." It is no longer a bet; it is a fact that needs action. An issue gets an *action owner* and a *resolution path*, not a probability and a mitigation. The defining test: a risk asks "what's our plan *if*"; an issue asks "what's our plan *now*."

- **Dependency** — a needed input that lies **outside the team's control**: another team's deliverable, a vendor's milestone, a decision from a governance board. It is tracked in RAID as a watched commitment because an unmet dependency is the most common *cause* of both risks and issues. Dependencies get their full treatment — dated, owned, provider-and-consumer contracts — in `dependencies-and-coordination.md`; here, the D is the RAID *entry* that keeps the dependency visible and review it like everything else.

Two distinctions carry most of the confusion and are worth stating as rules:

**Risk vs. issue — the probability test.** If probability < 1, it is a risk; if probability = 1, it is an issue. Logging a present problem as a "risk" ("there is a risk the build is broken" — no, the build *is* broken, that's an issue) lets a team feel it has a plan when it actually has an unaddressed fact. The reverse error — logging a future maybe as an "issue" — burns action-owner attention on something that may never occur. When a risk's probability reaches 1, it is no longer a risk; it has *converted* (see Risk → issue conversion).

**Assumption vs. risk — the verification test.** An assumption is a belief you haven't checked; a risk is a possibility you have. "We assume team B's service will be available" is an assumption until you ask — at which point either it's confirmed (assumption closed) or you learn it's uncertain ("team B's service might slip," now a tracked risk). The discipline is to validate the *load-bearing* assumptions early enough that a falsified one becomes a risk you can still manage rather than an issue you can only absorb.

## Risk exposure = probability × impact

A risk is scored on two axes and the product orders the register. Without a score, every risk looks equally urgent and prioritization collapses into whoever spoke last.

```
Exposure = Probability × Impact
```

The standard operational scale is 1–5 on each axis, giving an exposure of 1–25, banded:

| Probability | | Impact | |
|---|---|---|---|
| 1 | Rare (<10%) | 1 | Negligible — absorbed in normal work |
| 2 | Unlikely (~25%) | 2 | Minor — small schedule/cost hit |
| 3 | Possible (~50%) | 3 | Moderate — visible slip, recoverable |
| 4 | Likely (~70%) | 4 | Major — milestone at risk |
| 5 | Almost certain (>90%) | 5 | Severe — outcome/deadline at risk |

| Exposure band | Score | Treatment |
|---|---|---|
| **Low** | 1–6 | Accept-and-watch; review on cadence |
| **Medium** | 8–12 | Active response required; owner assigned a mitigation |
| **High** | 15–25 | Mitigate now; escalate per threshold; on the status report |

The score's job is *ordering*, not precision. A 5×5 = 25 outranks a 5×2 = 10, and the register sorts descending by exposure so the review always starts at the top of the pile. Treat impact as the dominant axis when you must choose: a low-probability, catastrophic-impact risk (the 1×5) usually deserves more attention than its raw product suggests, because you cannot recover from the tail — which is exactly why some teams weight impact or carry a separate "is this survivable" flag alongside the product.

**Qualitative vs. quantitative scoring.** The 1–5 ordinal scale above is *qualitative* — fast, good enough to order a register, and the right default for most teams. *Quantitative* scoring assigns real numbers — probability as a percentage, impact in currency or days — and computes **expected monetary value** (EMV = probability × cost). A risk with a 40% chance of a $50,000 schedule overrun carries an EMV of $20,000; if a mitigation costs $8,000 and halves the probability, it buys a $10,000 reduction in expected loss for $8,000 — a defensible trade you can show a sponsor, and the kind of arithmetic that justifies a contingency reserve. Summing the EMV across the register gives a defensible reserve figure rather than a guessed percentage. Use quantitative scoring when the stakes justify the effort and the numbers exist: a large program weighing a six-figure contingency reserve, a decision between two mitigations with different costs, anything feeding a formal business case. Don't fake precision: a "37% probability" invented to look rigorous is worse than an honest "Possible (3)." Quantitative risk modelling at program scale, and the formal techniques behind it, route to `/axiom-sdlc-engineering`.

**Proximity — the third dimension.** Exposure orders risks by *how bad*; proximity orders them by *how soon and how fast*. Two risks both scored 12 are not equal if one could fire next week and the other not until next quarter — the near one needs its response in motion now, the far one can be watched. Proximity (sometimes split into *time-to-impact* and *velocity*, how fast the risk goes from trigger to full consequence) is what makes escalation *timing* sensible: a high-exposure risk that is also imminent and fast-moving jumps the queue past an equally-scored risk that is distant and slow. Carry proximity as at least a coarse near/medium/far flag alongside exposure; without it, the register tells you what's dangerous but not what's *urgent*, and those are different questions.

## The register as a living document

The difference between a register that's *managed* and one that's a *relic* is entirely cadence. A managed register has these properties:

- **It is reviewed on a fixed cadence.** Weekly for an active project, at every governance board for a program. The review is a standing agenda item, not an event that happens when someone remembers.
- **Risks are re-scored as conditions change.** Probability and impact are not fixed at logging. As an integration date approaches, a "Possible (3)" dependency risk may rise to "Likely (4)"; as a mitigation lands, it may fall. The re-score is the heartbeat — a register whose scores never move is a register no one is reading.
- **Passed risks are closed.** A risk whose window has gone by (the integration happened, the API shipped) is marked closed with a one-line note. Closing matters: an open register cluttered with dead risks hides the live ones, and the act of closing is the proof the review happened.
- **New risks are added continuously.** Risks don't only arrive at kickoff. Every status review, every dependency slip, every falsified assumption is a source of new entries. A register whose newest entry is dated kickoff is, by definition, dead.

The diagnostic question for any RAID log: **when was the last time a risk's score changed, a risk was closed, or a risk was escalated?** If the answer is "kickoff," you have a graveyard (Anti-Pattern 1). A living register's audit trail shows movement — scores ticking up and down, entries opening and closing, escalations firing.

**What the review actually does.** "Review on cadence" is hollow without a procedure, so the RAID review runs a fixed loop each time, fast enough to fit a standing agenda slot: (1) walk the active register top-down by exposure; (2) for each, ask *has anything changed* — re-score probability and impact, and check whether the trigger has fired or moved closer; (3) confirm each response is still the right one and is actually being executed (a mitigation no one is doing is a fiction); (4) close anything whose window has passed; (5) sweep for *new* risks since last review — from status, from dependency slips, from falsified assumptions; (6) note anything that crossed an escalation threshold and route it. The whole loop is minutes, not an hour, because the active register is kept short (below). The output of every review is *movement*: scores changed, entries opened or closed, escalations queued. A review that produces no movement either caught a genuinely quiet week or — far more often — wasn't really a review.

**Active register vs. watch-list.** A register that tries to give equal review attention to every entry collapses under its own length — and a long register where everything is reviewed equally is how you end up with the "everything is medium" failure (Anti-Pattern 6). Split it. The **active register** holds the risks that warrant real attention each review — Medium and above, anything with a near proximity, anything whose trigger is close. The **watch-list** holds the Low-exposure, distant risks: logged so they aren't forgotten, but reviewed only periodically or when something changes, not line-by-line every week. The split keeps the review focused on the risks that can actually move the outcome while preserving the long tail. Entries graduate between the two as their scores move — a watch-list risk whose probability rises crosses onto the active register, and a mitigated active risk drops to the watch-list before it's closed. This is the practical mechanism behind "surface only the top exposures": the status report carries the active register's top entries, not the whole log.

## Risk responses: the four strategies

Every active risk gets a chosen response. There are four classic strategies for threats, and choosing one is a *decision*, recorded against the risk:

- **Avoid** — eliminate the cause so the risk cannot occur. Drop the feature that depends on the unproven vendor; change the architecture so the fragile integration isn't needed. Avoidance removes the risk entirely but usually costs scope or design freedom.
- **Reduce / Mitigate** — lower the probability, the impact, or both, without eliminating the risk. Build a fallback path (lowers impact), spike the risky integration early (lowers probability by buying information), add a buffer (lowers impact on the schedule). Most active risks are managed here.
- **Transfer** — move the risk to a party better placed to carry it. Insurance, a fixed-price contract that puts overrun risk on the supplier, an SLA with penalties. Transfer doesn't make the risk disappear; it relocates the *consequence*. (Note the residual: a transferred risk often leaves a smaller secondary risk — the counterparty failing to honour the transfer.)
- **Accept** — consciously decide to carry the risk, because mitigation costs more than the expected loss or because no response is available. Acceptance is legitimate and often correct. But it must be **explicit, owned, and documented, with a trigger condition** — never a silent decision to ignore. (See the discipline below; this is Anti-Pattern 2.)

**Inherent vs. residual exposure.** A response changes the score, and the register should record *both* numbers. The **inherent exposure** is the risk before any response — the raw probability × impact you face if you do nothing. The **residual exposure** is what remains *after* the chosen response is in place: a mitigation that halves probability turns an inherent 4×4 = 16 into a residual 2×4 = 8. The residual is the exposure you are actually carrying, and it is what you escalate and report on; the inherent is the justification for the response ("we spent this effort to drop a 16 to an 8"). Two failures hide when you track only one number: scoring only the inherent makes a well-mitigated risk look worse than it is and clutters the High band; scoring only the residual hides how much work is holding the risk down, so when the mitigation lapses no one realizes the exposure has snapped back to 16. A transfer leaves a residual too — the *secondary risk* that the counterparty fails to honour the transfer — and that residual is itself a tracked entry.

For **opportunities** (positive risks — a future event that might happen *in your favour*), the symmetric strategies are: **Exploit** (make it certain to happen), **Enhance** (raise its probability or benefit), **Share** (partner with someone who can help realize it), and **Accept** (take it if it comes, don't invest to chase it). Opportunity management is underused — registers skew entirely to threats — and a program that only logs what might go wrong never positions to catch what might go right.

**The acceptance discipline.** "Accept" is the most abused response because it looks like a decision while functioning as a dustbin. A real acceptance has three parts: a **named owner** who is accountable for the accepted risk, a **documented rationale** ("mitigation costs more than the expected loss"), and a **trigger condition** — a leading indicator that says "the thing we accepted is now becoming real, revisit the decision." An acceptance without a trigger is just a risk no one wanted to deal with, wearing a decision's clothing. If you cannot write the trigger, you have not actually accepted the risk — you have hidden it.

## Ownership and triggers

**Every risk has a named owner.** Not "the team," not "the PM by default" — a specific person accountable for watching the risk, executing its response, and raising it when it moves. An unowned risk is watched by no one, which is the same as not being tracked. The owner is usually the person best placed to see the trigger fire, not necessarily the most senior.

**Every risk should have a trigger** — a *leading indicator* that the risk is materializing, defined in advance while thinking is calm. "If the vendor's beta hasn't shipped by the end of month two" is a trigger; "if things look bad" is not. The trigger is what converts risk management from anxiety into a tripwire: you don't have to continuously worry about a risk whose trigger you've defined, because you've pre-committed to the action the trigger fires. Good triggers are *observable* (someone will actually notice them) and *early* (they fire with enough lead time to act). A trigger that only fires when the risk has already become an issue is useless — that's not a trigger, that's a post-mortem.

## The escalation path and thresholds

A risk escalates **before** it becomes an issue. That sentence is the whole point of the section: the entire reason to score and watch risks is to act while there is still a *future* to influence. Escalation that waits for the issue has failed.

Define escalation by two triggers, set in advance:

1. **Exposure threshold** — when a risk's re-scored exposure crosses a band boundary, it escalates to the next level of governance. A risk crossing into High (≥15) goes to the steering committee; a risk crossing into the survivable-tail zone (any impact-5) goes up regardless of probability.
2. **Trigger fired** — when a risk's leading indicator fires, it escalates immediately, regardless of its current score, because the trigger means the probability just jumped.

Escalation routes *along the stakeholder and governance map* — you escalate to the person with the authority and interest to act, which is why this depends on `stakeholder-and-communication.md` (who has power and interest) and `program-structure-and-governance.md` (the program risk board and decision rights). A worked threshold:

| Exposure / event | Owned by | Escalates to | Cadence |
|---|---|---|---|
| Low (1–6) | Team member | Not escalated | Team weekly review |
| Medium (8–12) | Delivery lead | Delivery lead's review | Weekly status |
| High (15–25) | Delivery lead | Steering committee / sponsor | Next governance board, or immediately if trigger fired |
| Any impact-5 (survivable-tail) | Delivery lead | Sponsor | Immediately on logging |
| Trigger fired (any score) | Risk owner | Per current band, one level up | Immediately |

The thresholds are set once and applied mechanically, so escalation isn't a judgement call made under pressure (when judgement is worst) but a rule that fires on its own. A risk that surprises a sponsor as a red issue is, almost always, a risk whose escalation threshold was never defined or never enforced (Anti-Pattern 3).

**Escalation is a request, not a confession.** The cultural failure underneath most dead escalation paths is that raising a risk is treated as admitting failure, so people sit on risks until they detonate. Counter it explicitly: an escalation is a *request for a decision or resource the owner can't supply at their level* — "this risk now needs a call only the steering committee can make," or "mitigating this needs budget I don't control." Framed that way, escalating early is the responsible act and *not* escalating is the failure. The escalation should arrive with the risk's score, its trigger status, the response already taken, and the specific decision or help being asked for — not just "here's a scary thing," but "here's a scored risk, here's what I've done, here's the decision I need from you." An escalation that names no ask is noise and trains the recipient to ignore the next one.

**What happens after escalation.** Escalation isn't the end state — it's a handoff that must close. The escalated risk gets a *decision* at the level it reached (accept the residual, fund a stronger mitigation, change scope to avoid it) and that decision is recorded back against the risk entry. A risk that escalates and then vanishes — no decision logged, no change in response — is worse than one never escalated, because now two levels believe the other is handling it. Close the loop: every escalation produces a recorded decision and a return of ownership to whoever now carries the (re-scored) risk.

## Risk → issue conversion

When a risk's probability reaches 1 — the event happened, the trigger fired and the thing it warned of arrived — the risk is no longer a risk. It **converts to an issue**:

1. The risk entry is marked *converted/closed*, with a pointer to the new issue (the audit trail must show the risk became this issue — don't silently delete it).
2. A new issue is opened: probability = 1, an **action owner** (who may differ from the risk owner — the issue needs whoever can *resolve* it), a **resolution path**, and a target resolution date.
3. The mitigation that was prepared for the risk (if any) becomes the opening move of the resolution.

After any significant issue, ask the **after-action question: was this in the register, and if not, why not?** Three answers, each with a different lesson:

- **It was in the register, scored, owned, with a fired trigger that escalated** → the system worked; the issue is being handled and was seen coming. This is success, even though there's an issue.
- **It was in the register but never re-scored or escalated** → the *log* worked but the *discipline* didn't; fix the cadence, not the log.
- **It was not in the register at all** → a *blind spot*; ask why the risk wasn't foreseen and whether a class of risk is being systematically missed (a missing assumption, a dependency no one mapped).

A mature team measures itself partly on how many of its issues were *previously tracked risks*. A high ratio of surprise issues (never in the register) means the register isn't looking far enough ahead.

## Managing issues, not just logging them

An issue is not a risk with the probability filled in to 1 — it's a different object that needs different fields, and treating it like a risk (mitigation, trigger, watching) is how issues fester. An issue's spine is:

- **Severity / priority.** Issues are ranked by *impact now* and *urgency*, not by an exposure product (probability is 1 for all of them). A severity scale — Critical (outcome or deadline at immediate risk) / High / Medium / Low — orders which issue the team works first, the same way exposure orders risks.
- **An action owner who can resolve it.** Not a watcher, not the person who logged it — the person who can actually close it. The most common issue-management failure is an issue "owned" by someone with no authority to fix it, so it sits open while everyone assumes someone else is on it.
- **A resolution path and a target date.** What specifically will close this, by when. An issue with no target date is an issue no one is driving; the date is what makes "still open" visible as a problem rather than a permanent state.
- **A resolution SLA by severity.** Critical issues get a same-day response and daily review until closed; lower severities get proportionally looser cadence. The SLA is what stops a High issue from quietly aging into the furniture.

Issues are reviewed at the *same* RAID cadence as risks but with the opposite question: not "is this becoming more likely?" (it's already certain) but "is this being resolved fast enough, and is the resolution actually working?" An issue that's been open across several reviews with no movement is itself a signal — either it's harder than scoped (re-plan it) or it's unowned in practice (re-assign it).

## Assumptions as latent risks

Assumptions deserve their own discipline because they are the quietest source of failure: a wrong assumption produces an issue with *no risk in between*, because no one was watching for it. Treat them as latent risks:

- **Log them.** Every "we're assuming…" that the plan rests on goes in the A column. The act of writing it down often exposes that it's shakier than it felt.
- **Validate the load-bearing ones.** Not every assumption needs checking — only the ones the plan would break without. For those, validate *early*, while a falsified assumption can still be managed as a risk rather than absorbed as an issue. The test of "load-bearing": if this turned out false, would we change the plan? If yes, validate it.
- **Convert on falsification.** A falsified assumption is never just deleted. If its consequence is still in the future, it becomes a *risk* (now scored and owned). If the consequence is already on you, it becomes an *issue* (action owner, resolution path). The conversion is the point — an assumption that's quietly proven wrong and left in the A column is a manufactured surprise.

## A worked lifecycle: one risk, end to end

The pieces above only matter when they connect. Trace a single risk through its whole life (generic illustration — a third-party API a feature depends on):

1. **Logged.** Early in delivery, the team records `R-014`: "The third-party API may not reach production readiness before our integration milestone, blocking the dependent feature." Inherent score: probability *Possible (3)*, impact *Major (4)* → exposure 12, Medium. Owner: the delivery lead, who talks to the vendor weekly. Response: *Reduce* — spike the integration against the vendor's beta early to buy information, and design a degraded fallback path (lowering impact). Trigger: "vendor's beta not shipped by the end of month two." Residual after the fallback design: 3×3 = 9, still Medium.

2. **Re-scored as the date nears.** At a later review the vendor's roadmap has slipped publicly; probability rises to *Likely (4)*. Residual exposure climbs to 4×3 = 12. Still Medium, but moving in the wrong direction — and the `last reviewed` date and the rising score are exactly what make the review notice it.

3. **Trigger fires.** End of month two arrives and the beta hasn't shipped. The trigger fires: per the escalation rule, the risk escalates *immediately* regardless of its band, because a fired trigger means the probability just jumped toward 1. It goes to the steering committee at the next governance touchpoint — *before* it has become an issue, which is the entire point.

4. **Converts to an issue.** Two weeks later the vendor confirms the API will not be ready for the milestone. Probability is now 1. `R-014` is marked *converted/closed* with a pointer to a new issue `I-009`: action owner is the engineer who can stand up the fallback path (not the delivery lead — the issue needs whoever can *resolve* it), resolution path is "ship the degraded path for the milestone, integrate the real API in the following increment," target date set. The fallback designed back in step 1 becomes the opening move — the mitigation paid off.

5. **After-action.** At the increment retrospective: *was this in the register?* Yes — scored, owned, with a trigger that fired and escalated on time. The system worked; the milestone took a known, planned hit instead of a surprise. The lesson logged is not "we failed" but "our vendor-readiness risk class is real; carry it on every external-dependency feature."

That is what a living RAID log buys: not the absence of bad outcomes, but bad outcomes that arrive *seen, planned-for, and owned* instead of as surprises.

## Keeping the register complete: a risk-category checklist

The hardest failure to fix is the risk that was never logged at all — the blind spot from the after-action question's third branch. The antidote is a **taxonomy**: a standing checklist of risk *categories* run against the work, so the question shifts from "what risks can I think of?" (which misses whole classes) to "do we have a risk in each category, and if not, are we sure?" Generic categories that catch most blind spots:

- **Technical** — unproven technology, performance/scalability unknowns, architectural bets not yet validated.
- **Schedule** — estimate uncertainty, milestone compression, critical-path fragility.
- **External / vendor** — third-party deliverables, API readiness, supplier or partner reliability (the category in the worked example).
- **Resource** — key-person dependency, skills gaps, contention with other work.
- **Scope / requirements** — drift, ambiguity, late-discovered requirements (managed in `scope-and-backlog-management.md`, surfaced here as risk).
- **Integration** — the seams between teams and systems where components meet (overlaps `dependencies-and-coordination.md`).
- **External environment** — regulatory, market, or organizational change outside the program's control.

Run the categories at kickoff to seed the register and re-run them periodically — a category that has *zero* entries is a prompt to check whether you're genuinely risk-free there or just not looking. The categories also make the register *legible*: grouping by category shows where exposure concentrates and which class of risk keeps converting to issues.

## Boundary: operational RAID vs. formal RSKM/DAR

This sheet is the **operational** RAID discipline: a working log, reviewed and escalated, scored well enough to order and act on. It is deliberately lean — for most teams, the discipline *is* the value, and a heavier process adds bureaucracy without adding safety.

**Regulated and high-maturity programs need more, and that lives in `/axiom-sdlc-engineering`.** Specifically:

- **Risk Management (RSKM)** as a defined process area — a documented risk management strategy, formal risk categories and parameters, quantitative risk thresholds tied to mitigation triggers, and an auditable risk-handling process — is the formal counterpart of this sheet's operational log. When a context demands *that the process itself be defined, repeatable, and audited* (safety-critical, regulated, contractually mandated), route to `/axiom-sdlc-engineering`'s governance-and-risk material.
- **Decision Analysis and Resolution (DAR)** — the formal, criteria-based evaluation of alternatives for significant decisions (which mitigation, which response, which contingency) — is the process area that backs the *decisions* this sheet makes informally. When a decision must be made against documented criteria with a recorded rationale and a defensible audit trail, that's DAR, and it routes to `/axiom-sdlc-engineering`.

The rule of thumb matches the pack spine: **`/axiom-sdlc-engineering` defines the risk and decision *process*; this sheet *runs* the risk management inside it.** A regulated program uses both — the formal RSKM/DAR procedures for the audit, this operational RAID discipline for the day-to-day management that executes them. Don't reach for the heavy process when the lean log is sufficient; don't pretend the lean log satisfies a regulator who requires the defined process area.

To **generate or refresh the artifact**, use the `/build-raid` command: it constructs a RAID log from the current state — each risk scored for exposure, each dependency made a dated owned commitment, with a review cadence and escalation thresholds wired in — producing a living artifact rather than a kickoff relic.

## A RAID entry schema

Every entry, whatever its type, carries a common spine so the register can be sorted, reviewed, and audited uniformly:

| Field | Meaning |
|---|---|
| **id** | Stable identifier (e.g. `R-014`, `I-003`, `A-007`, `D-011`) — the prefix encodes the type |
| **type** | Risk / Assumption / Issue / Dependency |
| **description** | One sentence; for risks, phrased as "*event* may happen, causing *consequence*" |
| **owner** | Named person accountable (for issues, the *action* owner who can resolve it) |
| **probability** | 1–5 for risks; n/a for issues (always 1) and dependencies; for assumptions, confidence it holds |
| **impact** | 1–5 — the consequence if it occurs |
| **exposure** | probability × impact (risks); track *inherent* (pre-response) and *residual* (post-response); residual drives ordering |
| **proximity** | how soon / how fast it could hit — near / medium / far; orders by urgency, not just severity |
| **response** | Avoid / Reduce / Transfer / Accept (risks); resolution path (issues); validation plan (assumptions) |
| **trigger** | Leading indicator that the risk is materializing / the assumption is failing |
| **status** | Open / Mitigating / Accepted / Escalated / Converted / Closed |
| **last reviewed** | Date of last re-score or review — the field that exposes a graveyard at a glance |

The `last reviewed` field is the cheapest liveness check in the pack: scan the column, and any date older than two review cycles is a risk no one is managing.

## Anti-Patterns

1. **The RAID log is a graveyard.** Risks are logged once at kickoff and never reviewed, re-scored, escalated, or closed. The log exists for an auditor or a process checkbox, not for the team; its newest entry is dated kickoff and no score has ever moved. This is the central failure — every other anti-pattern here is a symptom of it. *Fix: put the RAID review on a fixed cadence (weekly for a project, every governance board for a program) as a standing agenda item; re-score live; close passed risks; track `last reviewed` and treat any stale entry as a defect. A log that doesn't move is not being read.*

2. **"Accept" as a silent dustbin.** Risks no one wants to deal with are marked "Accepted" and forgotten — acceptance functioning as a way to make a risk stop demanding attention rather than as a conscious, owned decision. *Fix: require every acceptance to carry a named owner, a documented rationale, and a trigger condition that says "revisit this." If you can't write the trigger, you haven't accepted the risk — you've hidden it. Acceptances are reviewed like any other open entry.*

3. **No escalation path, so risks become surprise issues.** There is no defined threshold at which a risk goes up and to whom, so risks sit at the team level until they detonate as red issues that surprise the sponsor — who knew nothing because escalation depended on someone's judgement in the moment instead of a pre-set rule. *Fix: define exposure thresholds and trigger-fired rules in advance, mapped to the governance/stakeholder structure; escalate mechanically when a threshold is crossed. The goal is to escalate a risk* before *it becomes an issue.*

4. **Conflating risks and issues.** Present problems are logged as "risks" (so the team feels it has a plan when it has an unaddressed fact), or future maybes are logged as "issues" (burning action-owner attention on things that may never occur). *Fix: apply the probability test — probability < 1 is a risk, probability = 1 is an issue — and the verification test for assumptions. When a risk's probability hits 1, convert it to an issue rather than leaving it mislabelled.*

5. **Probability × impact scored once, never updated.** A risk is scored at logging and the score is treated as permanent, even as the integration date approaches (raising probability) or a mitigation lands (lowering it). The register's ordering goes stale and the review starts from the wrong top. *Fix: re-score at every review as conditions change; the moving score* is *the management. A score that never changes is the signature of a register no one is reading.*

6. **Every item is "medium."** The register is so long and so uniformly scored that nothing stands out — a hundred entries all banded Medium, which is the same as having no prioritization at all. *Fix: enforce the exposure distribution — most risks should be Low, a handful Medium, very few High; if everything is Medium, the scoring scale isn't being used honestly. Prune Low risks to a watch-list, surface only the top exposures on the status report, and make the High band genuinely scarce so it means something.*

## Cross-References

- `dependencies-and-coordination.md` — the D in RAID in depth: dependencies as dated, owned, provider-and-consumer commitments. This sheet tracks the dependency as a RAID entry; that sheet manages the seam that an unmet dependency turns into a risk or issue.
- `status-reporting-and-metrics.md` — the top risks (and any fired triggers) surface in the status report; this sheet's exposure ordering decides which risks make the report, and the report is where escalation often first becomes visible upward.
- `stakeholder-and-communication.md` — escalation routes along the stakeholder map (power/interest); who a risk escalates *to* is defined there, so escalation reaches the person with authority and interest to act.
- `program-structure-and-governance.md` — program-level risk governance: the risk board, decision rights, and the cadence at which program risks are reviewed and escalated above any single project.
- `/axiom-sdlc-engineering` — the **formal** Risk Management (RSKM) and Decision Analysis (DAR) process areas for regulated/high-maturity contexts: defined risk strategy, quantitative thresholds, criteria-based decision records, auditable process. That pack *defines* the risk and decision process; this sheet *runs* the operational risk management inside it. Route there when the process itself must be defined, repeatable, and audited.
- `/build-raid` — the command that generates or refreshes the RAID log as a living artifact: each risk scored for exposure, each dependency a dated owned commitment, review cadence and escalation thresholds wired in.
