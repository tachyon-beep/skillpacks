---
name: reliability-engineering
description: Use when a service falls over the moment a dependency slows down, when one struggling downstream takes the whole system with it, when retries pile onto an already-overloaded backend and turn a blip into an outage, when there is no error budget and "is this reliable enough" is an argument not a number, when on-call is drowning in repetitive manual work, when capacity is "we'll add servers when it's slow", when nobody has ever restored a backup, when there is no disaster-recovery plan and no stated RTO/RPO, when a region or AZ loss has never been rehearsed, or when load is rising and you cannot say how close to the cliff you are. Covers SRE error budgets and SLOs, toil reduction, capacity planning and load shedding, graceful degradation, retries with exponential backoff and jitter, circuit breakers and bulkheads, chaos engineering, disaster recovery with measured RTO/RPO, and backups that are proven by restore.
---

# Reliability Engineering

## The production stake

Reliability is not an attribute you add at the end — it is the assumption that *everything fails* and the discipline of failing in ways that don't take the whole system down or lose data you can't get back. Two classes of incident dominate the worst postmortems, and both are *self-inflicted*:

1. **Cascading failure.** One dependency gets slow. Callers block on it, exhaust their threads/connections, and become slow themselves. Their retries multiply load on the already-struggling dependency, which now fails completely. The failure propagates *upward* through every layer until a single slow database has taken down the entire product. Nothing "broke" — the system amplified a small problem into a total outage because it had no isolation, no backpressure, and no off-switch.
2. **The backup that was never a backup.** Data is lost — bad migration, ransomware, fat-fingered `DROP`. Someone reaches for the backups. The backup job has been "succeeding" for two years. The restore fails: the dumps are corrupt / incomplete / encrypted with a lost key / missing a table / take 40 hours when the business needed 4. An untested backup is not a backup; it is a *belief*. You find out which it was at the worst possible moment.

This sheet enforces one stance: **assume failure, bound its blast radius, and prove your recovery before you need it.** Reliability that has never been measured (no SLO), never been exercised (no chaos, no restore drill), and has no off-switch (no circuit breaker, no load shedding) is not reliability — it is luck that hasn't run out yet. This is engineering discipline, not a tool tour.

## Reliability is a number, not an adjective: SLO and error budget

You cannot manage what you don't measure, and "reliable" with no number is unfalsifiable. The SRE foundation is:

- **SLI** — a *Service Level Indicator*: a measured ratio of good events to total events (e.g. fraction of requests served < 300ms with a non-5xx). Measured from the **user's** vantage point, not the server's.
- **SLO** — a *Service Level Objective*: the target for that SLI over a window (e.g. 99.9% over 28 days). This is a deliberate choice, not 100% — 100% is the wrong target because it forbids all change and costs infinitely more than the marginal user values.
- **Error budget** — `1 − SLO`. At 99.9% over 28 days you may be "bad" for ~40 minutes. That budget is a *currency*: spend it on releases, experiments, and chaos. When it's exhausted, you freeze risky change and spend the next window on reliability work. The error budget converts "should we ship or harden?" from an argument into arithmetic.

```yaml
# Sloth SLO spec -> generates Prometheus recording + multi-window burn-rate alerts.
# Alert on BUDGET BURN RATE, not raw error rate: page fast when the budget is
# being consumed fast, ticket when it's a slow leak. This is the SRE alerting model.
version: prometheus/v1
service: orders-api
slos:
  - name: requests-availability
    objective: 99.9                 # SLO: 99.9% over 30 days -> ~43m/month error budget
    description: 99.9% of requests succeed
    sli:
      events:
        error_query: sum(rate(http_requests_total{job="orders-api",code=~"5.."}[{{.window}}]))
        total_query: sum(rate(http_requests_total{job="orders-api"}[{{.window}}]))
    alerting:
      name: OrdersApiHighErrorBudgetBurn
      page_alert:   { labels: { severity: page } }    # fast burn -> wake someone
      ticket_alert: { labels: { severity: ticket } }  # slow burn -> business hours
```

Without this, every reliability decision is vibes. With it, you have a budget to spend and a tripwire that fires when you're spending it too fast.

## Toil reduction: the work that should not be human

**Toil** is manual, repetitive, automatable, reactive operational work that scales linearly with the service and produces no lasting value — restarting a wedged process, manually expanding a disk, copy-pasting a runbook's commands every deploy. Toil is corrosive: it consumes the engineering time that *would* prevent the next incident, and it burns out on-call.

The SRE rule of thumb: **cap toil at ~50% of an SRE's time**; the rest goes to engineering that *eliminates* toil. The discipline:

1. **Measure it.** Tag operational tickets/pages as toil. If you don't track it, it silently eats 90% of the team.
2. **Rank by `frequency × time × people`.** Automate the top of that list first.
3. **Automate the response, not just the alert.** A runbook step that is always "run these three commands" should be a script the alert triggers (or an operator that self-heals). A page that a human resolves identically every time is an automation backlog item, not an on-call duty.
4. **A pager that fires for the same cause weekly is a defect, not a duty.** Fix the cause or automate the cure; do not normalize the interrupt.

Toil reduction is what makes the rest of this sheet sustainable — a team drowning in manual work never gets to capacity planning or restore drills.

## Resilience patterns: bounding the blast radius

These patterns exist for one reason: **stop a partial failure from becoming a total failure.** Apply them at every network call to a dependency that can be slow or down.

### Timeouts — the prerequisite for everything

A call with no timeout is the root of most cascades: it blocks a thread/connection indefinitely, those resources exhaust, and the caller dies waiting on a dependency that already died. **Every** remote call gets a timeout shorter than the caller's own deadline. No timeout is a latent outage.

### Retries — with exponential backoff *and jitter*

Retries fix transient blips but *cause* cascades when done naively: immediate, unbounded retries multiply load on a struggling dependency (the "retry storm"), and synchronized retries from many clients arrive in a thundering herd. The rules:

- **Backoff exponentially** so each retry gives the dependency more room.
- **Add jitter** so clients don't resynchronize into waves.
- **Cap attempts** (2–3, not infinite) and **only retry idempotent / retry-safe operations**.
- **Budget retries** — a token-bucket / retry-budget so the *aggregate* retry rate can't exceed a small fraction of base traffic. This is the single most important defense against retry-storm cascades.

```python
# tenacity: exponential backoff + jitter + capped attempts + retry only transient errors.
# The jitter and the cap are not optional polish -- they are what prevents the retry storm.
from tenacity import (
    retry, stop_after_attempt, wait_exponential_jitter,
    retry_if_exception_type, before_sleep_log,
)
import httpx, logging

log = logging.getLogger("payments")

@retry(
    stop=stop_after_attempt(3),                       # cap attempts; not infinite
    wait=wait_exponential_jitter(initial=0.2, max=5), # exp backoff + full jitter
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def charge(idempotency_key: str, amount_cents: int) -> dict:
    # Idempotency key makes the retry SAFE -- the server dedupes a double-charge.
    # Without it, retrying a non-idempotent write is how a blip becomes a double bill.
    resp = httpx.post(
        "https://payments.internal/charge",
        headers={"Idempotency-Key": idempotency_key},
        json={"amount_cents": amount_cents},
        timeout=httpx.Timeout(2.0, connect=0.5),       # ALWAYS a timeout
    )
    resp.raise_for_status()
    return resp.json()
```

### Circuit breakers — stop hammering the dead

When a dependency is failing, retrying it is *worse* than failing fast: you waste resources and prolong its outage. A circuit breaker watches the failure rate; when it crosses a threshold it **opens** — calls fail immediately (or fall back) without touching the dependency — then after a cooldown it goes **half-open** to probe recovery, and **closes** when the probe succeeds. This converts a slow, cascading failure into a fast, contained one and gives the dependency room to recover.

### Bulkheads — isolate so one failure can't sink the ship

Partition resources (connection pools, thread pools, concurrency limits) *per dependency* so that one slow dependency saturating its pool cannot starve calls to healthy dependencies. Named after ship compartments: a breach floods one, not all.

```go
// gobreaker: circuit breaker + an explicit bulkhead (bounded concurrency) around a dependency.
// The breaker stops hammering a dead dep; the semaphore stops a slow dep from eating ALL
// the caller's goroutines and starving every other dependency.
package recommendations

import (
	"context"
	"errors"
	"time"

	"github.com/sony/gobreaker/v2"
	"golang.org/x/sync/semaphore"
)

var (
	bulkhead = semaphore.NewWeighted(20) // at most 20 concurrent calls to THIS dependency

	breaker = gobreaker.NewCircuitBreaker[[]Rec](gobreaker.Settings{
		Name:        "recs-service",
		MaxRequests: 3,               // half-open probe budget
		Interval:    30 * time.Second,
		Timeout:     10 * time.Second, // cooldown before half-open
		ReadyToTrip: func(c gobreaker.Counts) bool {
			// Open when >=10 requests and >50% are failing.
			return c.Requests >= 10 &&
				float64(c.TotalFailures)/float64(c.Requests) > 0.5
		},
	})
)

func GetRecs(ctx context.Context, userID string) []Rec {
	// Bulkhead: refuse rather than queue unboundedly when the dep is slow.
	if err := bulkhead.Acquire(ctx, 1); err != nil {
		return fallbackRecs() // graceful degradation, not an error to the user
	}
	defer bulkhead.Release(1)

	recs, err := breaker.Execute(func() ([]Rec, error) {
		cctx, cancel := context.WithTimeout(ctx, 800*time.Millisecond) // always a timeout
		defer cancel()
		return fetchRecs(cctx, userID)
	})
	if err != nil {
		// Breaker open OR call failed -> serve a degraded-but-useful response.
		if errors.Is(err, gobreaker.ErrOpenState) {
			return cachedRecs(userID) // stale beats blank
		}
		return fallbackRecs()
	}
	return recs
}
```

### Graceful degradation and load shedding

Design every feature to have a *degraded mode* that is better than an error page: serve stale cache when the live source is down, hide the personalized panel when the recs service is open-circuited, queue the write when the async backend is saturated. The product should lose *features* under stress, not *availability*.

**Load shedding** is the system-level version: when overloaded, deliberately reject a fraction of work (cheapest/lowest-priority first) at the edge so the system serves the rest correctly, rather than accepting everything and collapsing into a state where it serves *nothing*. A server with no admission control under overload doesn't slow down gracefully — it falls over. Shed load before you hit the cliff.

## Capacity planning: know where the cliff is

"Add servers when it's slow" is how you discover your scaling limit during a traffic spike instead of in a load test. Capacity planning is the discipline of knowing your headroom *before* demand arrives:

1. **Load-test to find the knee.** Drive synthetic load (k6, Locust, Gatling) until latency/error rate breaks. That breakpoint — not the autoscaler's max — is your real ceiling per unit. Re-test after significant changes; capacity regresses silently.
2. **Model demand against it.** Project peak (launches, seasonal, marketing) and known growth. Maintain explicit **headroom** (commonly N+1 / N+2 and a utilization target well under the knee, e.g. plan to 60–70% of breakpoint) so a node loss or a spike doesn't immediately saturate.
3. **Autoscale for variance, provision for the floor.** HPA/cluster-autoscaler absorbs normal variation; pre-provision for known surges (autoscaling has lag and cold-start cost — it is not a substitute for capacity you *know* you'll need at a flash sale).
4. **Watch saturation as a leading signal.** The "saturation" of the four golden signals (latency, traffic, errors, saturation) tells you how close to the cliff you are *before* errors start. Alert on approaching saturation, not just on the outage it causes.

Capacity is also a reliability control: a system run at 95% utilization has no room to absorb a failure, a retry burst, or a chaos experiment.

## Chaos engineering: find the failure before it finds you

Resilience patterns you've never exercised are *hypotheses*. Chaos engineering is the discipline of injecting controlled failure into production-like (and eventually production) systems to verify that your timeouts, retries, breakers, fallbacks, and failover actually work — and to surface the cascade you didn't know you had.

The method is scientific, not "break things for fun":

1. **State a steady-state hypothesis** in terms of your SLI ("p99 stays < 300ms, error rate < 0.1%").
2. **Define the smallest blast radius** — one pod, one AZ, in staging first, then a bounded production experiment.
3. **Inject one failure** — kill a pod, add 200ms latency to a dependency, blackhole a downstream, fail a node.
4. **Measure against the hypothesis.** If steady state holds, your resilience is real. If it breaks, you found a latent outage *on your schedule* instead of at 3am.
5. **Always have an abort and automatic rollback.** A chaos experiment that you can't stop is just an outage.

```yaml
# Chaos Mesh (CNCF): inject 300ms latency into 50% of calls to the recs service for 5 min.
# Hypothesis: orders-api stays within SLO because the recs call has a timeout + breaker +
# cached fallback. If the SLO burns, the resilience is theatre and you just learned that safely.
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: recs-latency-experiment
  namespace: staging
spec:
  action: delay
  mode: fixed-percent
  value: "50"                      # bounded blast radius: half the targeted pods
  selector:
    namespaces: [staging]
    labelSelectors: { app: recs-service }
  delay:
    latency: "300ms"
    jitter: "50ms"
  duration: "5m"                   # auto-reverts -> built-in abort
```

Start in staging, run **game days** (rehearsed, scheduled failure exercises with the on-call team), graduate to small production experiments only once the basics hold. Chaos spends error budget deliberately — that's exactly what the budget is for.

## Disaster recovery: RTO, RPO, and the backup you've actually restored

DR is the plan for losing something big — a region, a database, a provider. It is defined by two numbers that the *business* sets, not engineering:

- **RTO (Recovery Time Objective)** — how long you may be down before recovery. Drives your failover architecture (hot standby vs. cold restore).
- **RPO (Recovery Point Objective)** — how much data you may lose, measured in time. Drives your replication/backup *frequency* (continuous replication for near-zero RPO; nightly dumps for a 24h RPO).

A DR plan that doesn't state RTO/RPO is not a plan — it's a wish. And RTO/RPO that have never been measured by a real drill are fiction.

**The backup discipline — non-negotiable:**

1. **3-2-1**: 3 copies, 2 media/locations, 1 off-site (and ideally one *immutable* / object-locked copy — ransomware deletes the backups it can reach).
2. **A backup is only proven by a restore.** Schedule **automated restore tests** that take a real backup, restore it to a scratch environment, and *verify the data* (row counts, checksums, a smoke query). A green backup job is not evidence; a green *restore* is.
3. **Measure the restore time** — that number *is* your real RTO. If the restore takes 40 hours and the RTO is 4, you don't have a DR plan, you have a discovery waiting to happen.
4. **Rehearse the regional failover** (game day) on a schedule. The first time you fail over should never be the real one.

```yaml
# GitHub Actions: nightly RESTORE TEST -- the only thing that proves a backup is real.
# Pulls the latest backup, restores to a throwaway DB, verifies, and MEASURES the restore
# time (your true RTO). Fails loudly (and pages) if restore or verification fails.
name: dr-restore-verification
on:
  schedule: [{ cron: "0 4 * * *" }]   # nightly
jobs:
  restore-test:
    runs-on: ubuntu-latest
    services:
      postgres: { image: postgres:17, env: { POSTGRES_PASSWORD: scratch }, ports: ["5432:5432"] }
    steps:
      - name: Fetch latest backup
        run: aws s3 cp "s3://prod-backups/$(date -u +%F)/orders.dump" ./orders.dump

      - name: Restore and time it (this elapsed time IS your RTO)
        id: restore
        run: |
          start=$(date +%s)
          pg_restore --clean --if-exists --no-owner \
            -d "postgresql://postgres:scratch@localhost:5432/postgres" ./orders.dump
          echo "seconds=$(( $(date +%s) - start ))" >> "$GITHUB_OUTPUT"

      - name: Verify the data is actually usable (not just that restore exited 0)
        run: |
          psql "postgresql://postgres:scratch@localhost:5432/postgres" -v ON_ERROR_STOP=1 -c \
            "DO \$\$ BEGIN
               IF (SELECT count(*) FROM orders) = 0 THEN
                 RAISE EXCEPTION 'restored orders table is empty -- backup is junk';
               END IF;
             END \$\$;"

      - name: Enforce RTO budget
        run: |
          if [ "${{ steps.restore.outputs.seconds }}" -gt 14400 ]; then   # 4h RTO
            echo "::error::Restore took ${{ steps.restore.outputs.seconds }}s, exceeds 4h RTO"
            exit 1
          fi
      # On failure: notify on-call. A silently-failing restore test is worse than none --
      # it manufactures false confidence.
```

## Common mistakes

- **No timeout on a remote call.** The seed of every cascade — a blocked thread waiting forever on a dead dependency, exhausting the pool. Every call gets a timeout shorter than the caller's deadline.
- **Naive retries (no backoff, no jitter, no budget, no cap).** Turns a transient blip into a retry storm that finishes off the struggling dependency. Exponential backoff + jitter + capped attempts + a retry budget, and only on idempotent ops.
- **Retrying non-idempotent writes.** A retried charge is a double charge. Use idempotency keys, or don't retry.
- **No circuit breaker.** You keep hammering a dead dependency, wasting resources and prolonging its outage. Break the circuit, fail fast, fall back.
- **No bulkheads / unbounded concurrency.** One slow dependency saturates the shared pool and starves every healthy one. Partition resources per dependency.
- **Accepting all load under overload.** No admission control means you serve nothing instead of degrading to serving most. Shed load and degrade features before the cliff.
- **SLO of "100%" or no SLO at all.** 100% forbids change and is infinitely expensive; no SLO makes reliability unfalsifiable. Pick a real target and run an error budget.
- **Alerting on raw error rate instead of burn rate.** Pages on every blip, misses slow leaks. Use multi-window burn-rate alerts against the budget.
- **Capacity = "add servers when slow."** You discover the ceiling during the spike. Load-test to the knee, plan headroom, watch saturation.
- **Backups that have never been restored.** The single most common catastrophic surprise. Automated restore tests with data verification, on a schedule, or you don't have backups.
- **DR plan with no RTO/RPO, never rehearsed.** Fiction. State the numbers, measure them with drills, fix the gap.
- **Toil normalized as on-call duty.** The same manual fix every week is an automation backlog item, not a job. Automate the cure or fix the cause.
- **Resilience patterns never exercised.** Untested timeouts/breakers/failover are hypotheses. Chaos-test them on your schedule, not the incident's.

## Red flags — STOP

If you catch yourself (or a teammate) saying any of these, stop and fix the discipline before proceeding:

- "We've never had to restore from backup." → Then you don't know if you can. Run a restore test today.
- "The backup job is green." → Green backup ≠ green restore. Only a verified restore counts.
- "We'll add a timeout if it becomes a problem." → The missing timeout *is* the problem; it's just latent. Add it now.
- "Just retry it until it works." → That's the retry storm that takes the dependency fully down. Backoff, jitter, cap, budget.
- "If a region goes down we'll figure it out." → Figuring it out live is the incident. Rehearse the failover before you need it.
- "We'll add capacity when it gets slow." → "Slow" is already past the knee. Load-test and plan headroom first.
- "100% uptime is the goal." → 100% forbids all change and costs infinitely. Set an SLO and an error budget.
- "We don't need a circuit breaker, the dependency is reliable." → Until the day it isn't, and you cascade. The breaker is for that day.
- "Chaos testing in production is reckless." → Untested resilience *is* the reckless state; chaos is the controlled, bounded, abortable alternative to discovering it at 3am.
- "On-call just restarts it, it's fine." → A weekly manual restart is a defect with a human patch. Automate the cure or fix the cause.

## Rationalizations and their counters

- **"We're too small for SRE practices."** Small teams have the *least* slack to absorb an outage and the *fewest* people who can recover one. A tested backup and a timeout cost a day; a lost database costs the company. Start with the cheap, high-leverage controls.
- **"Restore testing is a waste — backups always work."** They demonstrably do not: corrupt dumps, missing tables, lost keys, and restores that exceed RTO are the most common catastrophic surprises in the industry. The test is the only thing that converts belief into evidence.
- **"Resilience patterns add latency/complexity for failures that rarely happen."** Cascades are rare *and* catastrophic — the definition of what you insure against. A timeout and a breaker add microseconds on the happy path and save you a total outage on the bad day.
- **"Chaos engineering is for Netflix-scale companies."** The blast radius scales down: kill one staging pod, add latency to one dependency. The point is to learn your failure modes on your schedule. The smaller you are, the less you can afford to learn them during a real incident.
- **"Error budgets just slow us down."** They do the opposite: when the budget is healthy you ship *faster* with confidence because the data says you can. They only slow you when you're already burning reliability the users will feel.
- **"DR drills are theater — we have replication."** Replication faithfully replicates your bad migration and your ransomware to the standby. A drill is what proves you can actually cut over and serve, within RTO, from clean data.

## The bottom line

Assume everything fails. Put a number on "reliable" (SLO) and spend it deliberately (error budget). Bound every dependency call with a timeout, a capped jittered retry budget, a circuit breaker, and a bulkhead, so a partial failure degrades features instead of cascading into a total outage. Know where your capacity cliff is *before* the traffic arrives. Prove your resilience and your failover with chaos experiments and game days on your schedule. And treat a backup as worthless until a scheduled, automated restore has verified the data and measured the time against your RTO. Hold those and a dependency outage is a degraded feature and a restore is a routine drill. Skip them and you are running on luck — which is fine right up until the night it runs out.

## Cross-references

- `deployment-strategies` (this pack) — metric-driven automated rollback is a reliability control; canary analysis and error budgets share the same SLO telemetry.
- `ci-cd-pipeline-architecture` (this pack) — where restore-test and chaos jobs run as pipeline/scheduled gates.
- `infrastructure-as-code` (this pack) — codified, reproducible infra is the prerequisite for a rehearsable regional failover.
- Instrument vendor-neutrally with OpenTelemetry (OTLP) — the four golden signals and SLI queries feed both the burn-rate alerts and the chaos hypotheses.
- `/ordis-quality-engineering` — chaos engineering and resilience/load testing methodology in depth.
- `/axiom-solution-architect` — recording RTO/RPO targets and the DR architecture decision as an ADR.
