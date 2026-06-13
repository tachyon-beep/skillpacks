---
name: incident-response-and-oncall
description: Use when an outage is being fought over chat with nobody clearly in charge, when the same incident keeps recurring and no root-cause analysis was ever written, when one engineer is the only person who can fix prod and gets paged at 3am every week, when postmortems name a person instead of a cause or quietly never happen, when there are no severity levels so a cosmetic bug and a full outage get the same response, when on-call means thrashing through dashboards with no runbook, when "the deploy is frozen" has no defined trigger and stays frozen forever, when nobody can say whether you can afford to ship this week, or when the team brags about heroics while burning out. Covers severity classification, incident command, runbooks, incident comms and status updates, blameless postmortems and corrective-action tracking, SLOs and error budgets governing release pace, on-call rotation health and toil reduction.
---

# Incident Response and On-Call

## The production stake

The cost of an incident is not the failure — it is the *time spent failing*. Mean-time-to-recovery, not mean-time-between-failures, is what your users feel and your revenue records. A team with no incident discipline pays that cost in full every time: minutes bleed away while three engineers argue in a thread about what's happening, nobody owns the decision to roll back, the customer-facing status page says nothing, and the one person who understands the payments service is asleep and unpaged. The outage was thirty seconds of bad config; the *incident* was ninety minutes because the response was improvised.

The discipline this sheet enforces is that an incident is a **rehearsed operation with a single accountable commander, not an act of heroism.** When prod breaks, the questions "who is in charge," "how bad is this," "what do we tell customers," and "how do we get back to known-good" must already have answers — because inventing them mid-outage is where the minutes go. And after recovery, the failure must produce a **blameless root-cause analysis with tracked corrective actions**, or the same incident is queued to happen again. The two failures that end teams are *hero culture* (recovery depends on one irreplaceable person who eventually burns out or leaves) and *repeated incidents with no RCA* (the org keeps paying MTTR on a fault it already knows about).

This is engineering discipline, not a tool tour. PagerDuty, Opsgenie, and Statuspage are plumbing; the load-bearing parts are the **severity ladder, the command structure, the blameless postmortem, and the error budget that ties reliability to release pace.** Hold those and the tooling is interchangeable.

## Severity levels: the response is a function of severity

Before anything else, an incident needs a **severity**, because severity is what scales the response. Without a ladder, every blip triggers a war room (alert fatigue) or every outage gets the same lazy shrug (slow recovery). Severity is assigned on **customer impact**, not on how interesting the bug is or how loud the alert was.

| Sev | Meaning | Example | Response | Comms cadence |
|-----|---------|---------|----------|---------------|
| **SEV1** | Critical: major outage or data loss; core function unavailable to most users | Checkout down; auth broken; data corruption | Page IC + all relevant responders immediately; war room; exec notified | Status page now; updates every 30 min |
| **SEV2** | Major: significant degradation; a key feature down or a workaround exists | Search returns errors for 20% of users; elevated p99 breaching SLO | Page IC + on-call; investigate now | Status page if customer-visible; updates hourly |
| **SEV3** | Minor: limited impact; degraded but functional | One non-critical endpoint slow; cosmetic break | On-call handles in business hours; no war room | Internal ticket; no public status |
| **SEV4** | Low / informational | Flaky internal dashboard; near-miss | Ticket, batch with normal work | None |

Three rules that keep the ladder honest:

1. **Severity is set by impact and can be *upgraded* freely.** When in doubt, declare higher — it is cheap to downgrade a SEV2 to SEV3 and expensive to discover at minute 40 that your SEV3 was a SEV1 the whole time. Make declaring an incident a one-command, no-blame action.
2. **The severity determines the response, automatically.** SEV1 pages a commander and opens comms; SEV3 does not wake anyone. If everything pages, nothing pages — alert fatigue is how real SEV1s get missed.
3. **Time-based auto-escalation.** A SEV2 unacknowledged or unresolved past a threshold escalates to the next person and may bump severity. The rotation tool enforces this; humans forget under pressure.

## Incident command: one brain, defined roles

The single biggest MTTR win is **putting one person in charge.** The Incident Commander (IC) owns the *response*, not the *fix* — the IC coordinates, decides, and communicates so the responders can focus on the system. This is adapted from ICS (the emergency-services incident command system) and is what Google SRE, PagerDuty, and every mature on-call org run.

**Roles (one person each; on a small incident the IC may wear several hats, but the IC role is never skipped):**

- **Incident Commander (IC)** — the single decision-maker. Decides rollback vs. forward-fix, declares/changes severity, delegates, and is the one throat to choke for *coordination*. The IC does not type fixes; the IC runs the room.
- **Operations / Responders** — the hands on the system: diagnose, mitigate, execute the rollback. They report to the IC.
- **Communications Lead** — owns external and internal updates (status page, stakeholders, support) so responders are not interrupted to write tweets. On small incidents the IC absorbs this.
- **Scribe** — keeps the timeline: what was observed, what was tried, what happened, with timestamps. This is the raw material for the postmortem; without it the timeline is reconstructed from faulty memory.

**The command loop the IC runs, out loud, on a repeating cadence:** *What do we know? What's the impact (severity still right?)? What are we trying? Did it work? Who's doing what next?* The point is shared situational awareness — everyone hears the same state, so nobody duplicates work or operates on a stale picture.

**The IC's prime directive is mitigate first, diagnose later.** Stop the bleeding — roll back, fail over, flip the feature flag, drain the bad node — *before* hunting root cause. Root cause is a postmortem activity; recovery is the incident's only job. (See `deployment-strategies` for the rollback mechanisms the IC reaches for.)

## Runbooks: the page must point to an action

An alert that fires at 3am and says only "p99 latency high" hands the on-call engineer a research project under stress. A **runbook** turns that into an executable procedure: what this alert means, how to confirm it, the first mitigations to try, escalation path, and the dashboards/queries to look at. **Every paging alert links to a runbook**; an alert with no runbook is either not actionable (delete it — it is noise) or under-documented (write it).

Runbooks are versioned in the repo next to the service, reviewed like code, and — critically — **kept honest by being exercised.** A runbook that was right a year ago and never re-run during an incident is fiction. The discipline: when an incident reveals the runbook was wrong or missing a step, fixing the runbook is a corrective action in the postmortem.

```markdown
# Runbook: orders-api — elevated 5xx rate

**Alert:** OrdersApi5xxRateHigh (fires when 5xx > 1% for 5m)
**Severity guidance:** SEV2 if sustained > 5m; upgrade to SEV1 if checkout is affected.
**Owner:** #team-orders   **Escalation:** orders on-call → orders eng lead → platform on-call

## 1. Confirm impact (is this real and customer-facing?)
- Dashboard: https://grafana.example.com/d/orders-overview
- PromQL (current error ratio):
  sum(rate(http_requests_total{app="orders-api",code=~"5.."}[2m]))
  / sum(rate(http_requests_total{app="orders-api"}[2m]))
- Check the SLO burn-rate panel. If fast-burn alert is also firing, treat as SEV1.

## 2. Mitigate FIRST (do not diagnose before stopping the bleed)
- Recent deploy in last 30m? → roll back:  kubectl argo rollouts undo orders-api
- New feature flag enabled? → kill it:      flagctl disable orders-new-ranking
- Downstream (payments) failing? → confirm with its runbook; if so this is a SEV on payments, not orders.

## 3. Diagnose (only after impact is contained)
- Tail traces in the OTel backend filtered to orders-api, status=ERROR.
- Check recent dependency latency (DB, payments) in the service map.

## 4. Escalate if not mitigated in 15 min
- Page orders eng lead; open a SEV2 incident channel if not already open.
```

## Incident communications

During an incident, **silence is interpreted as "nobody is handling it."** Comms is a first-class workstream, owned by the Communications Lead (or the IC on small incidents), on a cadence set by severity (table above). Two audiences, different content:

- **External / customers** — a status page (e.g. Statuspage, Instatus, or a self-hosted page) with honest, non-technical, regular updates: *what is affected, that you are on it, when the next update comes.* You do not need root cause; you need presence. Commit to the *next-update time* and hit it even if the update is "still investigating."
- **Internal / stakeholders** — a dedicated incident channel (auto-created by the rotation tool), where the IC posts status on the command-loop cadence. Stakeholders read; they do not interrupt responders. The scribe's timeline lives here.

The cardinal comms rules: **never speculate publicly about cause or ETA** (you will be wrong and quoted on it), **acknowledge before you explain** (presence beats precision early), and **the IC owns the decision to publish**, so a responder's hot take doesn't become a public statement.

## Blameless postmortems: the failure must teach

An incident that does not produce a written, blameless postmortem is an incident you have *agreed to repeat.* The postmortem's job is to find the systemic causes and produce **tracked corrective actions** — not to assign fault. This is the single highest-leverage practice in the sheet, and the one most often skipped under "we're too busy" (which means "we're too busy to stop the next outage").

**Blameless is a mechanism, not a sentiment.** It rests on the assumption that everyone acted reasonably given the information they had at the time. The output of "person X ran the wrong command" is *firing person X and learning nothing*; the output of "the deploy tool let a destructive command run against prod with no confirmation, and the runbook didn't warn about it" is *two corrective actions that stop the whole class of failure.* The second is what blameless buys you: people speak honestly because they're not on trial, so you get the real causes. **Human error is a symptom of a system that allowed it, never the root cause.**

**Trigger criteria, written down:** every SEV1 and SEV2 gets a postmortem, full stop, regardless of how quickly it was fixed (a near-miss caught in 2 minutes still exposed a real fault). SEV3 at the IC's discretion. No postmortem is optional because the incident "wasn't that bad" — that's exactly how repeated incidents accumulate.

**Root-cause analysis beyond "the first cause."** Stop at the first plausible cause and you fix a symptom. Use Five Whys or a contributing-factors map to reach the systemic layer, then **address every contributing factor as a corrective action** — not just the trigger:

```markdown
# Postmortem: orders-api checkout outage — 2026-06-10

**Severity:** SEV1   **Duration:** 14:02–14:51 UTC (49 min)   **Author:** IC (J. Rivera)
**Impact:** ~38% of checkout attempts returned 5xx for 49 min. Est. 4,200 failed orders.

## Timeline (UTC) — from the incident scribe
- 14:02  Deploy of orders-api v2.31.0 reaches 100% (canary analysis passed on synthetic load).
- 14:05  5xx alert fires. On-call ack. No runbook step for "new release + DB pool exhaustion."
- 14:14  IC declared SEV1, war room opened, status page posted.
- 14:31  Mitigation: rolled back via `argo rollouts undo`. 5xx begins dropping.
- 14:51  Error rate back to baseline. Incident resolved; downgraded.

## Root-cause analysis (Five Whys → contributing factors)
1. Why did checkout 5xx?            DB connection pool exhausted under real traffic.
2. Why exhausted?                   v2.31.0 opened a connection per request (regressed pooling).
3. Why not caught pre-prod?         Canary analysis ran on synthetic load with low concurrency.
4. Why did canary pass?             AnalysisTemplate watched error rate, not pool saturation.
5. Why 49 min to recover?           No runbook entry for this signature; mitigation improvised.

## What went well
- Rollback path existed and worked (Argo Rollouts); blast radius bounded once identified.

## Corrective actions (each has an owner, a ticket, and a due date — tracked to done)
| # | Action | Type | Owner | Ticket | Due |
|---|--------|------|-------|--------|-----|
| 1 | Add DB pool-saturation metric to canary AnalysisTemplate | Prevent | M. Osei | OPS-1841 | 06-20 |
| 2 | Canary load test must hit realistic concurrency | Prevent | M. Osei | OPS-1842 | 06-27 |
| 3 | Add "5xx + pool exhaustion" runbook entry w/ rollback first | Mitigate | J. Rivera | OPS-1843 | 06-17 |
| 4 | Lint rule: reject per-request connection creation | Prevent | platform | OPS-1844 | 07-04 |

## Notes
Blameless: no individual is named as cause. The deploy was approved through the normal,
reviewed process; the gap was in what our automated gates measured, not in anyone's judgment.
```

**Corrective actions are not done when written — they are done when *done.*** The most common postmortem failure is a beautiful document whose action items rot in a backlog. Track them as real tickets with owners and due dates, review open postmortem actions in a recurring forum, and treat an aging corrective action as the open invitation to the *next* incident that it is. A postmortem with no completed actions did not happen.

## Error budgets: reliability governs release pace

This is where reliability stops being a vibe and becomes a number that **controls how fast you ship.** You set a **Service Level Objective** (e.g. 99.9% of requests succeed over 28 days). The complement — `1 − SLO`, here 0.1% — is your **error budget**: the amount of unreliability you are *allowed* to spend. Downtime, failed requests, and SLO-breaching latency all draw it down.

The error budget converts the eternal dev-vs-ops fight ("ship faster" vs "stop breaking prod") into a **shared, automatic policy:**

- **Budget remaining → ship.** You are within your reliability target; feature velocity is the priority. Take deployment risk; that's what the budget is for.
- **Budget exhausted → freeze.** The release freeze is **triggered by the budget, not by argument or seniority.** Until the budget recovers, the team works reliability — corrective actions, hardening, paying down the debt that burned the budget. This is the antidote to "we keep shipping through outages because a VP wants the feature."
- **Burn-rate alerting** pages on the *rate* of consumption, not just the absolute level: a **fast-burn** alert (e.g. consuming 2% of the 28-day budget in 1 hour) is a SEV-worthy page now; a **slow-burn** alert is a ticket. This is how you alert on user-facing SLO impact instead of on every CPU twitch.

```yaml
# Prometheus multi-window, multi-burn-rate SLO alerts (the Google SRE workbook pattern).
# SLO: 99.9% success over 28d  =>  error budget = 0.1%.
# Fast burn: short window AND long window both hot => page. Slow burn => ticket.
groups:
  - name: orders-api-slo
    rules:
      # Fast burn: 14.4x budget burn => budget gone in ~2 days. Page immediately.
      - alert: OrdersErrorBudgetFastBurn
        expr: |
          (
            sum(rate(http_requests_total{app="orders-api",code=~"5.."}[5m]))
            / sum(rate(http_requests_total{app="orders-api"}[5m]))
          ) > (14.4 * 0.001)
          and
          (
            sum(rate(http_requests_total{app="orders-api",code=~"5.."}[1h]))
            / sum(rate(http_requests_total{app="orders-api"}[1h]))
          ) > (14.4 * 0.001)
        labels: { severity: page, slo: orders-availability }
        annotations:
          summary: "orders-api burning error budget 14.4x — fast burn"
          runbook: "https://runbooks.example.com/orders-api/slo-fast-burn"
      # Slow burn: 3x over 6h/long window => budget threatened but not urgent. Ticket.
      - alert: OrdersErrorBudgetSlowBurn
        expr: |
          (
            sum(rate(http_requests_total{app="orders-api",code=~"5.."}[6h]))
            / sum(rate(http_requests_total{app="orders-api"}[6h]))
          ) > (3 * 0.001)
        labels: { severity: ticket, slo: orders-availability }
        annotations:
          summary: "orders-api slow error-budget burn — investigate this sprint"
```

The SLO metrics here are queried from a **vendor-neutral OpenTelemetry** pipeline (OTLP → Collector → Prometheus or equivalent) — instrument the service once, against the OTel SDK, never a vendor agent, so the same metrics feed alerting, canary analysis (`deployment-strategies`), and the postmortem timeline.

## On-call health and toil

A rotation that burns people out is a reliability risk: exhausted responders make worse decisions, and attrition takes irreplaceable context with it. On-call is a system to be **engineered for sustainability**, with measured load.

- **Toil is the enemy and it is measurable.** Toil = manual, repetitive, automatable operational work that scales with traffic and produces no lasting value (the SRE definition). Track the fraction of on-call time spent on toil; when it's high, the corrective action is *automation*, not *more on-call*. A pager that fires for a known, scripted remediation should fire at the script, not the human.
- **Page only on actionable, user-impacting signals.** Every page should be (a) urgent, (b) actionable, and (c) tied to user impact — ideally an SLO burn-rate alert, not a raw resource threshold. Non-actionable alerts are deleted or demoted to tickets/dashboards. **Alert fatigue from noisy paging is how real incidents get acked-and-ignored.** Review page volume per shift as a health metric; a healthy rotation has *few* pages.
- **Sustainable structure.** Follow-the-sun across time zones where possible so nobody is routinely woken; primary + secondary (backup) so a missed page still escalates; **compensate on-call** (time off or pay) — unpaid expectation of 24/7 availability is how you lose senior people. Bound shift length and frequency.
- **Reduce single points of failure — the direct antidote to hero culture.** If one person is the only one who can resolve a class of incident, that is a *bug in the team*, not a feature of that person. Runbooks, pairing on incidents, rotating who holds the pager, and game-days/chaos drills spread the knowledge. The goal is a rotation where *any* on-call engineer can run *any* incident from the runbooks — no name is load-bearing.

## Common mistakes

- **Hero culture.** Recovery depends on one irreplaceable person who is paged for everything and praised for the saves. It feels great until they burn out, go on vacation during a SEV1, or quit — taking the only copy of the recovery knowledge with them. The fix is mechanical: runbooks, shared rotation, postmortem corrective actions that *transfer* the knowledge out of the hero's head. Celebrate the boring rotation where anyone can handle anything, not the 3am save.
- **Repeated incidents with no RCA.** The same fault recurs because the team keeps *recovering* without ever *learning* — no postmortem, or a postmortem whose corrective actions were never completed. You are paying MTTR on a known bug, on repeat. Mandate postmortems for SEV1/SEV2 and track corrective actions to *done*.
- **No incident commander.** Everyone debugs in parallel, nobody decides, two people roll back conflicting things. One IC, deciding and coordinating (not typing fixes), is the biggest MTTR lever there is.
- **Blameful postmortems.** Naming a person as "the cause" guarantees the next person hides the truth, and you fix nothing systemic. Human error is the symptom; the system that permitted it is the cause.
- **Postmortem theater.** A perfect document whose action items rot in a backlog. The writeup is worthless without *completed* corrective actions tracked like real work.
- **No severity ladder.** Either everything pages (alert fatigue, real SEV1s missed) or nothing does (slow recovery). Severity, set by impact, scales the response.
- **Diagnosing before mitigating.** Hunting root cause while users are down. Stop the bleed first — roll back, fail over, kill the flag — *then* investigate in the postmortem.
- **Alerts with no runbook.** A page that hands the on-call a research project at 3am. Every paging alert links to an actionable runbook, or it's deleted as noise.
- **No error budget / freeze ignored.** Reliability is a vibe nobody can enforce, so the team ships through outages because someone senior wants the feature. An error budget makes the freeze *automatic and impersonal* — triggered by the number, not the argument.
- **Resource-threshold paging instead of SLO burn-rate.** Paging on 80% CPU wakes people for non-events and misses user-facing failures that don't move CPU. Alert on what users feel: SLO burn rate.
- **Silent incidents.** No status page, no updates, so customers assume you don't know or don't care. Acknowledge fast, update on a cadence, hit your next-update commitment.

## Related sheets

- `deployment-strategies` — the rollback/failover/feature-flag mechanisms the IC reaches for to *mitigate first*; most incidents are changes, so fast revert is the primary recovery tool.
- `ci-cd-pipeline-architecture` — the build-once, signed-artifact pipeline whose gates the error budget governs; a frozen budget freezes the release pipeline.
- `infrastructure-as-code` — recovery from infra loss depends on environments being rebuildable from versioned code, not reconstructed from memory under incident pressure.
- For the vendor-neutral telemetry powering SLO alerts, burn-rate calculation, and postmortem timelines, instrument with OpenTelemetry (OTLP → Collector) — never a vendor SDK.
