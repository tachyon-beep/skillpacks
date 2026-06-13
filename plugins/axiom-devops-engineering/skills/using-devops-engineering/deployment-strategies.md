---
name: deployment-strategies
description: Use when a deploy ships all instances at once with no way back, when "deploy to production" is a single manual step, when a bad release means an outage until someone reverts and rebuilds, when there is no traffic-shifting or canary and every change hits 100% of users instantly, when rollback means "redeploy the old tag and pray", when you cannot answer "how do we undo this in under a minute", when a schema change and the code that needs it ship together, or when you are choosing between blue-green, canary, rolling, and feature-flag dark launches. Covers zero-downtime deployment strategies, per-strategy verification and automated rollback triggers, progressive delivery with Argo Rollouts and Flagger, Gateway API traffic-shifting, expand/contract migrations, and killing the big-bang deploy.
---

# Deployment Strategies

## The production stake

A deploy is the single most dangerous routine operation you perform. Most outages are not novel bugs — they are *changes*, pushed to 100% of traffic, with no rehearsed way back. The big-bang deploy (stop, swap, start, hope) trades minutes of convenience for an open-ended outage when the new version is wrong, because the only recovery path is "notice, diagnose, rebuild the old artifact, redeploy" while users are down.

The discipline this sheet enforces is simple and non-negotiable: **every production deploy ships behind a strategy that (a) keeps the old version serving until the new one is proven, (b) limits blast radius while it proves itself, and (c) has an automated, metric-driven trigger that reverts faster than a human can be paged.** If you cannot answer "what flips us back, and how long does it take," you do not have a deployment strategy — you have a big-bang deploy with extra steps.

This is engineering discipline, not a tool tour. The strategies below are mechanisms for one goal: *make rollback cheaper and faster than forward-fixing.*

## The decision: which strategy

Pick by what the failure costs and what you can observe — not by what is fashionable.

| Strategy | Rollback speed | Blast radius before proof | Infra cost | Needs good metrics? | Best for |
|----------|---------------|---------------------------|-----------|---------------------|----------|
| **Rolling** | Slow (re-roll instances) | One batch at a time | 1x (+surge) | No | Internal tools, low-stakes stateless services, the K8s default |
| **Blue-green** | Instant (flip traffic) | All-or-nothing at cutover | 2x during deploy | Smoke tests suffice | Critical systems where instant revert matters more than gradual exposure |
| **Canary** | Fast (route to stable) | A small % of real traffic | ~1.1x | **Yes** — automated rollback is metric-driven | High-traffic services with real SLO telemetry |
| **Feature flag / dark launch** | Instant (toggle, no deploy) | Per-user / per-cohort | 1x | Yes for guarded rollout | Decoupling *release* from *deploy*; risky features; A/B; kill-switches |

**How to choose, in order:**

1. **Can you observe success automatically?** No reliable error-rate / latency / SLO signal → you cannot run a *metric-gated* canary safely. Use blue-green (smoke-test gate) or rolling with strong health checks until you have telemetry. Do not run a "canary" that no automation watches — that is just a slow big-bang.
2. **Is instant, total revert the priority?** (payments, auth, anything where 30 seconds of badness is unacceptable) → **blue-green**. One traffic flip back to the known-good environment.
3. **Is the service high-traffic with good SLO data, and is gradual risk exposure worth the complexity?** → **canary** via Argo Rollouts or Flagger.
4. **Do you want to ship code now but turn it on later, or for a subset of users?** → **feature flags / dark launch**, layered *on top of* whichever rollout strategy moves the binary.
5. **Otherwise** (cost-sensitive, lower stakes) → **rolling** with proper readiness gates and `maxUnavailable: 0`.

These compose. The mature pattern is: rolling/blue-green to move the *binary* safely + feature flags to control *exposure* of the behavior + canary analysis to *gate progression*. They are not mutually exclusive.

## Strategy mechanics

### Rolling

Replace instances in batches; the orchestrator keeps the rest serving. This is Kubernetes' default Deployment behavior. It is the cheapest zero-downtime option but the slowest to roll back, because reverting means re-rolling every updated instance.

The two settings that make rolling safe — and that teams routinely get wrong:

- **`maxUnavailable: 0`** — never take healthy capacity *down* to make room; surge up first. The default (`25%`) can drop a quarter of your fleet mid-deploy.
- **A readiness probe that actually means "ready to serve"** — checks dependencies, not just "process is up." A rolling deploy with a lying readiness probe sends traffic to broken pods and is indistinguishable from a big-bang.

```yaml
# k8s Deployment — safe rolling config
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-api
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0      # never drop healthy capacity
      maxSurge: 2            # add new pods before removing old
  template:
    spec:
      terminationGracePeriodSeconds: 45   # finish in-flight requests
      containers:
        - name: orders-api
          image: registry.example.com/orders-api@sha256:<digest>  # pin by DIGEST, never :latest
          readinessProbe:                 # gates traffic — must check real readiness
            httpGet: { path: /health/ready, port: 8080 }
            periodSeconds: 5
            failureThreshold: 3
          lifecycle:
            preStop:
              exec: { command: ["sleep", "15"] }   # let LB deregister before SIGTERM
```

Rollback: `kubectl rollout undo deploy/orders-api`. It works, but it is a *second* rolling update — minutes, not seconds. If you need faster, you need blue-green or canary.

### Blue-green

Two complete environments. **Blue** serves production. You deploy the new version to **green**, run smoke tests against it while it takes zero production traffic, then flip the router. Rollback is flipping the router back — effectively instant. Cost is ~2x during the overlap.

The failure mode blue-green *closes* is "I clicked deploy and now I'm stuck." The cost is doubled infra during cutover and the discipline of keeping blue warm long enough to revert.

```yaml
# Argo Rollouts — blue-green, with an analysis gate and manual promotion
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: payments-api
spec:
  replicas: 4
  strategy:
    blueGreen:
      activeService: payments-active        # serves prod traffic
      previewService: payments-preview      # green, for smoke tests pre-flip
      autoPromotionEnabled: false           # require explicit/automated promotion
      prePromotionAnalysis:                 # MUST pass before traffic flips
        templates: [{ templateName: smoke-and-error-rate }]
      postPromotionAnalysis:                # watches AFTER flip; abort -> auto flip back
        templates: [{ templateName: error-rate-slo }]
      scaleDownDelaySeconds: 600            # keep blue warm 10m for instant rollback
  selector:
    matchLabels: { app: payments-api }
  template:
    metadata: { labels: { app: payments-api } }
    spec:
      containers:
        - name: payments-api
          image: registry.example.com/payments-api@sha256:<digest>
```

Rollback trigger: `postPromotionAnalysis` failing aborts the rollout and Argo flips `activeService` back to blue automatically. Manual escape hatch: `kubectl argo rollouts undo payments-api`.

### Canary

Send a small slice of *real* production traffic to the new version, watch SLO metrics, and progress only if they hold. This gives the lowest blast radius before proof — but it is **only as safe as the metrics that gate it.** A canary with no automated analysis is theater. The rollback is routing 100% back to stable, which is fast.

Modern canary uses **Gateway API** for traffic-shifting (Ingress-NGINX retired March 2026; Gateway API v1.5 is the traffic standard, implemented by Envoy Gateway, Istio, Cilium, Kong). Argo Rollouts and Flagger both drive Gateway API `HTTPRoute` weights.

```yaml
# Argo Rollouts — canary with metric-driven automated rollback via Gateway API
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: search-api
spec:
  replicas: 10
  strategy:
    canary:
      trafficRouting:
        plugins:
          argoproj-labs/gatewayAPI:
            namespace: search
            httpRoute: search-route        # Gateway API HTTPRoute, not Ingress
      steps:
        - setWeight: 5
        - pause: { duration: 5m }
        - analysis:                        # gate: bad metrics -> abort -> route 100% to stable
            templates: [{ templateName: canary-slo }]
        - setWeight: 25
        - pause: { duration: 10m }
        - setWeight: 50
        - pause: { duration: 10m }
        - setWeight: 100
  template:
    spec:
      containers:
        - name: search-api
          image: registry.example.com/search-api@sha256:<digest>
---
# AnalysisTemplate — the automated rollback trigger.
# successCondition failing => Rollout aborts => traffic returns to stable. No human in the loop.
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: canary-slo
spec:
  metrics:
    - name: error-rate
      interval: 1m
      count: 5
      failureLimit: 1                      # one bad reading aborts
      successCondition: result < 0.01      # < 1% 5xx on the canary
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{app="search-api",version="canary",code=~"5.."}[2m]))
            /
            sum(rate(http_requests_total{app="search-api",version="canary"}[2m]))
    - name: p99-latency
      interval: 1m
      count: 5
      failureLimit: 1
      successCondition: result < 0.5       # p99 < 500ms
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{app="search-api",version="canary"}[2m])) by (le))
```

**Argo Rollouts vs Flagger** — same goal, different model. Argo CD dominates GitOps (~60% of clusters; 97% of its users in production) and pairs naturally with **Argo Rollouts** (a `Rollout` CRD replaces your `Deployment`). **Flux** users pair with **Flagger** (Flagger keeps a normal `Deployment` and drives the canary externally). Pick by your GitOps ecosystem, not by feature checklist — both do Gateway API / service-mesh traffic-shifting with metric-driven automated rollback. Anchor either to the OpenGitOps principles: declarative, versioned, pulled, continuously reconciled.

### Feature flags / dark launch

The other strategies move the *binary* safely. Feature flags decouple **deploy** (binary is in production) from **release** (behavior is on for users). The code ships dark, behind a flag defaulted off, and you turn it on for 1% → internal users → a cohort → everyone — with a **kill-switch** that reverts instantly *without a deploy*.

This closes failure modes the binary-rollout strategies cannot: a bad feature you cannot un-ship without a rebuild, a feature you want to A/B, a risky path you want a one-click off-switch for. **Dark launch** is the read-path variant: run the new code path against real traffic but discard its output (or shadow-compare), so you load-test and correctness-check in production before any user sees a result.

```python
# Server-side flag check. The branch is the rollback path — flip the flag, no deploy.
# (OpenFeature-style provider so you are not locked to one vendor SDK.)
from openfeature import api

client = api.get_client()

def list_orders(user):
    ctx = {"targetingKey": user.id, "plan": user.plan, "region": user.region}
    if client.get_boolean_value("orders-new-ranking", default=False, evaluation_context=ctx):
        return rank_orders_v2(user)      # dark-launchable: shadow-compare before exposing
    return rank_orders_v1(user)
```

Discipline that keeps flags from becoming a liability: flags default **off** and **fail closed** (provider unreachable → old behavior); every flag has an owner and a removal ticket (flags are debt — remove them once a feature is fully rolled out); kill-switch flags are evaluated on the hot path and documented in the runbook. A flag system *is* a rollback mechanism — treat its availability as production-critical.

## Verification and automated rollback — per strategy

A strategy without an automated revert trigger is a big-bang deploy that takes longer. For each strategy, define the **gate** (what must be true to proceed) and the **trigger** (what reverts automatically).

| Strategy | Verification gate (proceed if…) | Automated rollback trigger | Revert action |
|----------|-------------------------------|----------------------------|---------------|
| Rolling | Readiness probe passes per batch; no error spike | Probe failures exceed threshold; `progressDeadlineSeconds` exceeded | `rollout undo` (re-roll prior) |
| Blue-green | pre-promotion smoke + error-rate analysis pass on green | post-promotion analysis fails | flip router to blue (instant) |
| Canary | per-step `AnalysisTemplate` (error rate, p99, saturation) holds | any analysis metric breaches `successCondition` | route 100% to stable |
| Feature flag | guarded-rollout cohort metrics hold | flag-level metric guard breaches; manual kill-switch | toggle flag off (no deploy) |

Three rules that make these real:

1. **The trigger is automated, or it does not exist.** "We'll watch Grafana and revert if it looks bad" is not a rollback trigger — it is a hope. Wire the metric query into the rollout controller (`AnalysisTemplate` / Flagger metric checks).
2. **Health checks gate *traffic*, not *startup*.** A readiness probe must reflect the ability to serve real requests (dependencies reachable, caches warm), not "the process started." Liveness restarts a wedged pod; readiness controls whether it gets traffic. Confusing them ships traffic to broken instances.
3. **Pin artifacts by digest and build once.** The same immutable image (by `sha256` digest, never a mutable tag) flows staging → canary → prod. If staging and prod can resolve `:latest` to different bytes, your verification proved nothing. Sign the image *and* its SBOM with Sigstore/cosign (keyless via Fulcio + Rekor) and verify in-pipeline — a rollback to "the previous tag" is worthless if you cannot prove which bytes that was.

## Database migrations: the rollback trap

Most "we couldn't roll back" incidents are schema, not code. If the new code and a destructive schema change ship together, rolling back the code leaves it pointed at a schema it no longer understands — so you *can't* roll back. **Never couple a destructive migration to the deploy that needs it.** Use expand/contract (a.k.a. parallel change), spread across separate deploys:

```sql
-- DEPLOY 1 — EXPAND: additive only. Old code still works. Fully rollback-safe.
ALTER TABLE orders ADD COLUMN customer_ref text;        -- nullable, no default backfill lock
CREATE INDEX CONCURRENTLY idx_orders_customer_ref ON orders (customer_ref);  -- non-blocking

-- DEPLOY 2 — code writes BOTH old and new columns; backfill runs in batches; reads still old.
-- DEPLOY 3 — code reads NEW column (old still populated as fallback). Verify in prod.
-- DEPLOY 4 — CONTRACT: only after weeks stable, drop the old column.
ALTER TABLE orders DROP COLUMN legacy_customer_id;       -- irreversible; do LAST, alone
```

Each deploy is independently rollback-safe because at every step the running code works against the schema actually present. The contract step is the only irreversible one and it ships alone, long after the code that obsoleted the old column.

## Common mistakes

- **Big-bang with no rollback path.** Stop-swap-start to 100%, recovery = rebuild-and-redeploy under outage. This is the failure this sheet exists to kill. If there is no fast, rehearsed revert, you have not deployed safely — you have gambled.
- **"Canary" with no automated analysis.** Shifting 5% of traffic and eyeballing a dashboard is a slow big-bang. The metric gate must be wired into the controller (`AnalysisTemplate` / Flagger) so it reverts without a human.
- **Readiness probe that only checks "process up."** Traffic flows to instances that can't reach the DB. Probes must reflect ability to *serve*, and must be distinct from liveness.
- **Coupling destructive migrations to the deploy.** Drops the rollback path entirely — code revert lands on a schema it can't use. Always expand/contract across deploys.
- **Mutable tags (`:latest`) through the pipeline.** Staging and prod resolve to different bytes; verification is meaningless and rollback targets are ambiguous. Pin by digest, build once, sign and verify.
- **Forgetting connection draining.** Rolling/blue-green without `preStop` + grace period kills in-flight requests on every pod swap — zero-downtime on paper, 5xx spikes in practice.
- **Tearing down blue too fast.** `scaleDownDelaySeconds: 0` means your instant-rollback environment is gone the moment you flip. Keep it warm long enough to revert.
- **Flags that fail open or never get removed.** A flag that defaults on when the provider is unreachable is an outage amplifier; a flag never removed is permanent dead-branch debt. Default off, fail closed, owner + removal ticket per flag.
- **No post-promotion watch.** Gating *before* the flip but not *after* misses failures that only appear under full production load. Always run post-promotion analysis with an auto-revert.

## Related sheets

- `ci-cd-pipeline-architecture` — the surrounding pipeline: build-once artifacts, test gates, environment promotion, where these strategies plug in.
- For the metric backend powering canary analysis, instrument vendor-neutrally with OpenTelemetry (OTLP) and expose SLO metrics the `AnalysisTemplate` can query.
