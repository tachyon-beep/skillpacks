---
name: observability-and-monitoring
description: Use when an incident is "the site is slow" with no data to localize it, when the first signal of an outage is a customer tweet, when a deploy goes out and nobody watches anything, when on-call is buried under pages that auto-resolve and nobody reads them anymore, when alerts fire on CPU and disk but never on user-facing failure, when you cannot answer "what is our MTTR" because nothing measures it, when a dashboard has forty panels and none tell you if users are okay, when logs/metrics/traces live in three tools that don't correlate, when you cannot follow one request across services, or when you are picking between Prometheus, OpenTelemetry, and a vendor SDK. Covers the three pillars + OpenTelemetry/OTLP, SLI/SLO/error-budget design, RED and USE methods, symptom-based actionable alerting, dashboards that answer "are users okay", and killing alert fatigue, unmonitored deploys, and unmeasurable MTTR.
---

# Observability and Monitoring

## The production stake

The clock that decides whether an incident is a blip or a headline starts the moment something breaks and stops the moment you understand it. That interval — time-to-detect plus time-to-localize — is the part of MTTR you actually control, and observability is the only thing that compresses it. A system you cannot see fails the same way every time: it breaks, nobody notices until a customer complains, then three engineers spend forty minutes arguing over which service is at fault because no signal points at the culprit. The outage was five minutes of badness and fifty-five minutes of blindness.

The discipline this sheet enforces: **every production service emits the three signals (metrics, logs, traces) through one vendor-neutral pipeline; every user-facing service has SLIs measuring what users experience and SLOs with an error budget; every alert fires on a symptom a user feels, is actionable, and is rare enough that on-call still reads it; and every deploy is watched by something that can tell whether it made users worse.** If you cannot answer "are users okay right now," "is this deploy safe," and "what is our MTTR," you do not have observability — you have logs you grep after the fact.

This is engineering discipline, not a tool tour. The goal is not "have a dashboard"; it is *reduce the time between failure and understanding, and stop crying wolf so loudly that nobody comes when it's real.*

## The distinction that the whole sheet rests on

**Monitoring** answers questions you knew to ask: is CPU high, is the disk full, is the process up. **Observability** is the property that lets you answer questions you did *not* anticipate — "why are checkouts from EU mobile clients on the new payment provider 4x slower since 14:20?" — without shipping new code to find out. You get there by emitting high-cardinality, correlated, structured telemetry, not by adding more pre-defined gauges.

Monitoring is necessary (you still want a disk-full alert) but insufficient. Most teams over-invest in monitoring host metrics and under-invest in the user-experience SLIs and the trace context that actually localize a novel failure. This sheet pushes the second.

## The three pillars, and the one wire that carries them

| Signal | Answers | Cost / cardinality | Primary use |
|--------|---------|--------------------|-------------|
| **Metrics** | "Is something wrong, and how bad, over time?" | Cheap, aggregatable, low cardinality | Alerting, SLOs, dashboards, trends |
| **Traces** | "Where in the request path did it go wrong?" | Per-request, sampled | Localizing latency/errors across services |
| **Logs** | "What exactly happened at this point?" | Expensive at volume, high cardinality | Forensic detail, the *why* once a trace points you at the *where* |

The three only earn their keep when they **correlate**: a metric spikes → you pivot to exemplar traces for that spike → a slow span's `trace_id` joins you to the structured logs for that exact request. Three tools that cannot share a `trace_id` are three silos, and you will pay the correlation cost manually, under incident pressure, every time.

**OpenTelemetry (OTel) is the standard that makes correlation the default.** It is CNCF-graduated; traces, metrics, and logs are stable across SDKs; **Profiles is the fourth signal** (RC / stabilizing through Q1 2026). One wire protocol — **OTLP** — carries all signals. One semantic-convention layer means `http.request.method`, `service.name`, etc. mean the same thing everywhere. The **OTel Collector** (receivers → processors → exporters) sits between your apps and your backends so the apps never know or care which vendor stores the data.

**The non-negotiable rule: instrument with OpenTelemetry, never a vendor SDK.** A vendor agent in your code is a re-instrumentation project the day you change backends, and it fragments correlation. Instrument once with OTel; point the Collector wherever you want; switch backends by editing the Collector config, not your application. (For high-volume pipelines, **OTel Arrow** compresses OTLP; **eBPF** gives zero-code auto-instrumentation for a baseline before you add manual spans.)

## SLIs, SLOs, and the error budget — the spine of the whole practice

You cannot alert well, prioritize reliability work, or judge a deploy without first defining *what "okay" means to a user, numerically.* That is the SLI/SLO loop, and it is the single highest-leverage thing on this sheet.

- **SLI (Service Level Indicator)** — a ratio of good events to valid events, measured at the point closest to the user. `good / valid`, e.g. *fraction of HTTP requests served in <300ms with a non-5xx status*. SLIs are about user experience, not machine health. "CPU < 80%" is not an SLI; users do not feel CPU.
- **SLO (Service Level Objective)** — the target for an SLI over a window. *99.9% of requests succeed within 300ms over 28 rolling days.* An SLO is a deliberate choice that 100% is the wrong target — chasing the last fraction of a nine costs exponentially more and buys reliability users cannot perceive.
- **Error budget** — `1 − SLO`. At 99.9% you may "spend" 0.1% of requests failing — about 43 minutes per 30 days. This is the budget that makes reliability a *decision* instead of an argument: budget remaining → ship features faster; budget exhausted → freeze risky changes and spend the next cycle on reliability. The error budget is the contract between "move fast" and "stay up."

Choosing SLIs: pick the few that map to the user's actual goal. For a request-driven service the canonical set is **availability** (fraction of requests not failing), **latency** (fraction under a threshold), and where relevant **quality/correctness** and **freshness** (for data pipelines, "data no older than N minutes"). Measure them at the edge the user touches (load balancer / gateway), not deep inside, where you'd miss failures that never reached your code.

### Example 1 — SLO with multi-window, multi-burn-rate alerting (Prometheus)

The crude SLO alert — "page when the 28-day error rate crosses 0.1%" — fires far too late (the budget is already blown) and is jittery on short windows. The Google SRE-standard fix is **multi-window, multi-burn-rate**: page fast when the budget is burning fast, page slowly (or ticket) when it is burning slowly, and require a short *and* a long window to agree so a momentary blip doesn't page.

```yaml
# prometheus-rules.yaml — recording rules + multi-burn-rate SLO alerts
# SLI = fraction of good requests. SLO = 99.9% over 28d  => error budget = 0.1%.
groups:
  - name: checkout-slo
    rules:
      # Record the error ratio over several windows once, reuse in alerts.
      - record: job:slo_errors:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout",code=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="checkout"}[5m]))
      - record: job:slo_errors:ratio_rate1h
        expr: |
          sum(rate(http_requests_total{job="checkout",code=~"5.."}[1h]))
          / sum(rate(http_requests_total{job="checkout"}[1h]))
      - record: job:slo_errors:ratio_rate6h
        expr: |
          sum(rate(http_requests_total{job="checkout",code=~"5.."}[6h]))
          / sum(rate(http_requests_total{job="checkout"}[6h]))

      # FAST burn: 14.4x budget burn => exhausts 28d budget in ~2 days.
      # PAGE. Short(5m) AND long(1h) windows must agree to avoid flapping.
      - alert: CheckoutErrorBudgetFastBurn
        expr: |
          job:slo_errors:ratio_rate5m > (14.4 * 0.001)
          and
          job:slo_errors:ratio_rate1h > (14.4 * 0.001)
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Checkout burning error budget 14.4x — fast"
          # Symptom-based: users' checkouts are failing right now.
          description: "5xx ratio {{ $value | humanizePercentage }} on checkout. Budget exhausts in ~2d at this rate."
          runbook: "https://runbooks.example.com/checkout-availability"

      # SLOW burn: 6x => exhausts in ~3.3 days. TICKET, don't page at 3am.
      - alert: CheckoutErrorBudgetSlowBurn
        expr: |
          job:slo_errors:ratio_rate6h > (6 * 0.001)
          and
          job:slo_errors:ratio_rate1h > (6 * 0.001)
        for: 15m
        labels: { severity: ticket }
        annotations:
          summary: "Checkout burning error budget 6x — slow"
          description: "Sustained elevated 5xx on checkout. Investigate within business hours."
          runbook: "https://runbooks.example.com/checkout-availability"
```

Two properties make this *actionable* rather than noisy: it alerts on **how fast you are losing the user-facing budget** (a symptom), and the fast/slow split routes a real, urgent failure to a page and a slow simmer to a ticket — so the page that wakes someone is always worth waking for.

## RED and USE — what to measure, so you don't measure everything

Two complementary frameworks stop "what metrics do we need" from becoming "instrument everything and drown."

- **RED** — for **request-driven services** (APIs, web, anything serving requests). Per service, measure:
  - **Rate** — requests per second.
  - **Errors** — failed requests per second.
  - **Duration** — latency distribution (percentiles, not averages — averages hide the tail where the pain is).
  RED is symptom-oriented: it is exactly the surface users experience, and it maps straight onto availability and latency SLIs.

- **USE** — for **resources** (CPU, memory, disk, network, queues, connection pools). Per resource, measure:
  - **Utilization** — % time busy.
  - **Saturation** — queued/waiting work the resource can't service yet (the early-warning signal; saturation rises before utilization pins).
  - **Errors** — error events on the resource.
  USE is cause-oriented: it tells you *why* RED is bad once RED says something is.

The discipline: **alert on RED (symptoms users feel), investigate with USE (causes).** Most alert-fatigue disasters are USE metrics promoted to pages — "CPU 90%" wakes someone, but 90% CPU with the SLO green is *fine*, that's a well-utilized box. Page on the symptom; keep the cause metrics on the dashboard for when the symptom fires.

## Actionable alerting — alert on symptoms, not causes

This is where most observability programs quietly fail. Every alert must pass three tests, all of them:

1. **Symptom, not cause.** Page on "users' requests are failing / are slow / the SLO is burning," not on "CPU is high," "a pod restarted," "the queue has 1,000 items." Causes are legion and most are self-healing or harmless; symptoms are what the user feels. A cause-based alert that fires while the SLO is green is, by definition, not worth a page.
2. **Actionable.** When it fires, there is something a human must *do now*. If the response is "watch it" or "it'll clear up," it is not a page — at most it's a dashboard annotation. Every paging alert links a runbook with the diagnosis-and-mitigation steps.
3. **Urgent.** A page interrupts a human's life. Reserve it for "user-facing, happening now, needs intervention." Everything else is a ticket or a dashboard. The fast/slow burn split above is exactly this triage encoded.

**Alert fatigue is a production risk, not an annoyance.** When on-call gets twenty pages a shift, they stop reading them, and the twenty-first — the real one — gets acknowledged and ignored. Every non-actionable page actively degrades your incident response. The cure is ruthless: each alert must justify its existence against the three tests, and any alert that fired in the last quarter and required no action gets deleted or demoted. Track your **page-to-action ratio**; if most pages led to "no action needed," your alerting is broken regardless of how good your dashboards look.

## Watching deploys — closing the unmonitored-deploy gap

Most outages are changes (see `deployment-strategies`). An unwatched deploy is the single most common way a controllable incident becomes an uncontrolled one: the bad version ships, MTTD is "however long until a customer notices," and by then it's at 100%. Observability closes this two ways:

- **Annotate every deploy on your dashboards** (deploy markers / events). The first question in any incident is "what changed," and a vertical line on the latency graph at 14:20 answers it in one second instead of forty minutes of "did anyone deploy?"
- **Gate progressive rollout on SLI metrics.** This is where this sheet meets `deployment-strategies`: the canary's `AnalysisTemplate` queries the *same* RED/SLO metrics defined here, and a breach reverts traffic automatically. The metrics you build for SLOs are the metrics that make automated rollback possible. If a deploy can degrade the SLI without anything noticing, you have an unmonitored deploy regardless of how nice the pipeline is.

## Dashboards that answer "are users okay"

A dashboard with forty panels is a dashboard nobody reads in an incident. Structure for the question being asked, top to bottom:

- **Top: the SLO/user-experience row.** RED metrics and SLO burn — "are users okay right now," answerable in five seconds. This is what on-call opens first.
- **Middle: per-dependency RED.** Which downstream (DB, payment provider, each upstream service) is contributing the errors/latency. This is the localization layer.
- **Bottom: USE / resource metrics.** Causes, for when the top says something's wrong and you're chasing why.

Rules: percentiles not averages (p50/p95/p99 — the tail is where users hurt); every panel earns its place by answering a question someone asks in an incident; deploy markers on the time axis. A dashboard is for *navigating from symptom to cause fast*, not for displaying everything you happen to collect.

## Example 2 — vendor-neutral instrumentation with the OpenTelemetry Collector

Instrument the application once with the OTel SDK (auto-instrumentation gets you RED for free; add manual spans for business operations), export OTLP to a Collector, and let the Collector fan out. Switching backends is a Collector-config edit, never a code change.

```python
# app.py — instrument with OTel, export OTLP. No vendor SDK anywhere.
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# service.name is a semantic-convention attribute — it joins traces, metrics, logs.
resource = Resource.create({"service.name": "checkout", "service.version": "2.7.0"})

trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
metrics.set_meter_provider(MeterProvider(
    resource=resource,
    metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://otel-collector:4317"))],
))

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
checkout_latency = meter.create_histogram(
    "checkout.duration", unit="s", description="end-to-end checkout latency")

def checkout(cart, user):
    # A manual span for the business operation; auto-instrumentation covers HTTP/DB spans.
    with tracer.start_as_current_span("checkout") as span:
        span.set_attribute("cart.item_count", len(cart.items))
        span.set_attribute("user.tier", user.tier)   # high-cardinality dim for slicing later
        import time; t0 = time.monotonic()
        try:
            result = process_payment(cart, user)      # child spans auto-created downstream
            span.set_attribute("checkout.outcome", "ok")
            return result
        except PaymentError as e:
            span.record_exception(e)
            span.set_attribute("checkout.outcome", "payment_failed")
            raise
        finally:
            checkout_latency.record(time.monotonic() - t0, {"user.tier": user.tier})
```

```yaml
# otel-collector.yaml — receivers -> processors -> exporters.
# Swap backends HERE, never in the application.
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }

processors:
  batch: {}                              # batch before export (throughput)
  memory_limiter:                        # protect the Collector from OOM under load
    check_interval: 1s
    limit_percentage: 80
    spike_limit_percentage: 25
  tail_sampling:                         # keep all errors + slow traces, sample the boring ones
    decision_wait: 10s
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: keep-slow
        type: latency
        latency: { threshold_ms: 500 }
      - name: sample-rest
        type: probabilistic
        probabilistic: { sampling_percentage: 10 }

exporters:
  prometheus:                            # metrics backend
    endpoint: 0.0.0.0:8889
  otlp/traces:                           # trace backend (Tempo/Jaeger/any OTLP sink)
    endpoint: tempo:4317
    tls: { insecure: true }

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, tail_sampling, batch]
      exporters: [otlp/traces]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
```

The payoff: the application knows only "emit OTLP to the Collector." Backends, sampling policy, batching, and PII scrubbing all live in Collector config under version control. The day you change observability vendors, you edit one YAML file and redeploy the Collector — the instrumented services never change. That is what "vendor-neutral" buys you, and it is why a vendor SDK in app code is a trap.

## Common mistakes

- **Alerting on causes, not symptoms.** Paging on CPU/memory/pod-restarts/queue-depth fills on-call with noise that's often harmless. Page on RED/SLO symptoms; keep cause metrics on the dashboard.
- **Averages instead of percentiles.** A 200ms average can hide a p99 of 4s — the tail your worst-affected users live in. Always p95/p99 for latency SLIs.
- **SLOs measured deep inside the system.** Measuring availability at the app layer misses failures that never reached your code (LB 503s, gateway timeouts). Measure at the edge the user touches.
- **Instrumenting with a vendor SDK.** Locks you in and fragments correlation. Instrument with OTel; route via the Collector.
- **Three uncorrelated tools.** Metrics, logs, traces in silos with no shared `trace_id` means you correlate by hand under pressure. Propagate trace context; emit structured logs carrying `trace_id`.
- **No deploy annotations.** "What changed?" — the first incident question — takes forty minutes instead of one second. Mark every deploy on the dashboards.
- **100% as the reliability target.** Chasing the last nine costs exponentially for reliability users can't perceive. Set an SLO < 100% and run an error budget.
- **Unbounded log volume as the strategy.** "Log everything and grep later" is expensive, slow to query, and useless for detection. Logs are forensic detail *after* a metric/trace points you at the where.
- **High-cardinality labels on metrics.** Putting `user_id`/`request_id` on Prometheus labels explodes cardinality and can take the metrics system down. High cardinality belongs on traces and structured logs, not metric labels.
- **Dashboards built to display, not to navigate.** Forty panels nobody reads in an incident. Structure top-down: user-experience → dependencies → resources.
- **No runbook on the page.** A 3am page with no link to "here's what to check and do" wastes the most expensive minutes you have. Every paging alert links a runbook.

## Red flags — STOP

If you catch yourself or a teammate saying any of these, stop and fix the discipline before proceeding:

- "We'll know if it breaks — someone will notice." → That "someone" is a customer, and that interval is your MTTD. Define an SLI and alert on it.
- "Just add a CPU alert." → That's a cause alert that pages on healthy boxes. Alert on the symptom users feel.
- "We don't need SLOs, we just keep it up." → "Up" without a number is unfalsifiable. You can't run an error budget, gate a deploy, or prioritize reliability work without an SLO.
- "Log everything, we'll figure it out later." → Unbounded logs are slow to query and useless for *detecting* anything. Detection is metrics; logs are the forensic last mile.
- "Use the vendor's agent, it's one line." → That one line is a re-instrumentation project at switch time and a correlation silo. Instrument with OTel.
- "On-call ignores most pages anyway." → That is the diagnosis, not a workaround. Every ignored page is training on-call to ignore the real one. Delete or demote it.
- "Just deploy, we'll watch Grafana." → Nobody watches Grafana for forty minutes after every deploy. Annotate the deploy and gate the rollout on the SLI.
- "Put the user_id on the metric so we can slice it." → That's a cardinality bomb. User-level slicing lives on traces and logs.
- "We can't measure MTTR, incidents are all different." → If you can't measure it you can't improve it. Timestamp detect/mitigate/resolve on every incident.

## Rationalizations and their counters

- **"Observability is overkill for our scale."** Small teams have *fewer* eyes and *less* slack to absorb a blind outage. The smaller you are, the more you need detection to do the noticing for you.
- **"We have logs, that's enough."** Logs tell you what happened at one point *after* you already know where to look. They don't detect, they don't trend, and they don't localize across services. You need metrics to detect and traces to localize.
- **"SLOs are a Google thing for huge systems."** An SLO is just "what does 'okay' mean, numerically." Any service with users has one whether you wrote it down or not; writing it down is what lets you alert and prioritize.
- **"We'll set up alerting after launch."** Launch is exactly when you're most likely to break and least able to detect it by feel. The SLI/alert is part of "done," not a follow-up.
- **"More alerts = safer."** The opposite. Past the point on-call can read them, each added alert lowers the chance the real one gets actioned. Fewer, symptom-based, actionable alerts are safer than many.
- **"OpenTelemetry is more work than the vendor SDK."** More work today, far less work at every backend change and every cross-service correlation forever after. The vendor SDK's convenience is a loan against your future mobility.
- **"We can't afford the trace/log volume."** You don't keep all of it — tail-sample to keep every error and slow trace and a fraction of the boring ones (see the Collector config). The signal is in the failures; the cost is in the success you can drop.

## The bottom line

You control the part of MTTR between failure and understanding, and observability is the lever on it. Instrument once with OpenTelemetry through a Collector so the three signals correlate and you're never locked to a vendor. Define SLIs at the user's edge, set SLOs below 100%, and run the error budget as the contract between speed and stability. Alert only on symptoms users feel, only when a human must act, and rarely enough that on-call still reads the page. Annotate every deploy and gate rollouts on the SLIs. Hold those, and an incident is five minutes of badness and a fast diagnosis. Break them, and it's five minutes of badness wrapped in fifty-five minutes of blindness — and you find out from a customer.

## Cross-references

- `deployment-strategies` (this pack) — the canary `AnalysisTemplate` queries the RED/SLO metrics defined here; SLI-gated rollout with automated revert is the payoff of this instrumentation.
- `ci-cd-pipeline-architecture` (this pack) — where deploy annotations and SLO gates plug into the pipeline.
- `infrastructure-as-code` (this pack) — provision the Collector, backends, and alerting rules as code, not by hand.
- `/ordis-quality-engineering` — chaos engineering and performance testing exercise the observability you build here; verify the alerts actually fire.
- `/axiom-solution-architect` — record the OTel-vs-vendor and SLO-target decisions as ADRs.
