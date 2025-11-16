---
name: observability-and-monitoring
description: Use when implementing metrics/logs/traces, defining SLIs/SLOs, designing alerts, choosing observability tools, debugging alert fatigue, or optimizing observability costs - provides SRE frameworks, anti-patterns, and implementation patterns
---

# Observability and Monitoring

## Overview

**Core principle:** Measure what users care about, alert on symptoms not causes, make alerts actionable.

**Rule:** Observability without actionability is just expensive logging.

**Already have observability tools (CloudWatch, Datadog, etc.)?** Optimize what you have first. Most observability problems are usage/process issues, not tooling. Implement SLIs/SLOs, clean up alerts, add runbooks with existing tools. Migrate only if you hit concrete tool limitations (cost, features, multi-cloud). Tool migration is expensive - make sure it solves a real problem.

## Getting Started Decision Tree

| Team Size | Scale | Starting Point | Tools |
|-----------|-------|----------------|-------|
| 1-5 engineers | <10 services | Metrics + logs | Prometheus + Grafana + Loki |
| 5-20 engineers | 10-50 services | Metrics + logs + basic traces | Add Jaeger, OpenTelemetry |
| 20+ engineers | 50+ services | Full observability + SLOs | Managed platform (Datadog, Grafana Cloud) |

**First step:** Implement metrics with OpenTelemetry + Prometheus

**Why this order:** Metrics give you fastest time-to-value (detect issues), logs help debug (understand what happened), traces solve complex distributed problems (debug cross-service issues)

## Three Pillars Quick Reference

### Metrics (Quantitative, aggregated)

**When to use:** Alerting, dashboards, trend analysis

**What to collect:**
- **RED method** (services): Rate, Errors, Duration
- **USE method** (resources): Utilization, Saturation, Errors
- **Four Golden Signals**: Latency, traffic, errors, saturation

**Implementation:**
```python
# OpenTelemetry metrics
from opentelemetry import metrics

meter = metrics.get_meter(__name__)
request_counter = meter.create_counter(
    "http_requests_total",
    description="Total HTTP requests"
)
request_duration = meter.create_histogram(
    "http_request_duration_seconds",
    description="HTTP request duration"
)

# Instrument request
request_counter.add(1, {"method": "GET", "endpoint": "/api/users"})
request_duration.record(duration, {"method": "GET", "endpoint": "/api/users"})
```

### Logs (Discrete events)

**When to use:** Debugging, audit trails, error investigation

**Best practices:**
- Structured logging (JSON)
- Include correlation IDs
- Don't log sensitive data (PII, secrets)

**Implementation:**
```python
import structlog

log = structlog.get_logger()
log.info(
    "user_login",
    user_id=user_id,
    correlation_id=correlation_id,
    ip_address=ip,
    duration_ms=duration
)
```

### Traces (Request flows)

**When to use:** Debugging distributed systems, latency investigation

**Implementation:**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order.id", order_id)
    span.set_attribute("user.id", user_id)
    # Process order logic
```

## Anti-Patterns Catalog

### ❌ Vanity Metrics
**Symptom:** Tracking metrics that look impressive but don't inform decisions

**Why bad:** Wastes resources, distracts from actionable metrics

**Fix:** Only collect metrics that answer "should I page someone?" or inform business decisions

```python
# ❌ Bad - vanity metric
total_requests_all_time_counter.inc()

# ✅ Good - actionable metric
request_error_rate.labels(service="api", endpoint="/users").observe(error_rate)
```

---

### ❌ Alert on Everything
**Symptom:** Hundreds of alerts per day, team ignores most of them

**Why bad:** Alert fatigue, real issues get missed, on-call burnout

**Fix:** Alert only on user-impacting symptoms that require immediate action

**Test:** "If this alert fires at 2am, should someone wake up to fix it?" If no, it's not an alert.

---

### ❌ No Runbooks
**Symptom:** Alerts fire with no guidance on how to respond

**Why bad:** Increased MTTR, inconsistent responses, on-call stress

**Fix:** Every alert must link to a runbook with investigation steps

```yaml
# ✅ Good alert with runbook
alert: HighErrorRate
annotations:
  summary: "Error rate >5% on {{$labels.service}}"
  description: "Current: {{$value}}%"
  runbook: "https://wiki.company.com/runbooks/high-error-rate"
```

---

### ❌ Cardinality Explosion
**Symptom:** Metrics with unbounded labels (user IDs, timestamps, UUIDs) cause storage/performance issues

**Why bad:** Expensive storage, slow queries, potential system failure

**Fix:** Use fixed cardinality labels, aggregate high-cardinality dimensions

```python
# ❌ Bad - unbounded cardinality
request_counter.labels(user_id=user_id).inc()  # Millions of unique series

# ✅ Good - bounded cardinality
request_counter.labels(user_type="premium", region="us-east").inc()
```

---

### ❌ Missing Correlation IDs
**Symptom:** Can't trace requests across services, debugging takes hours

**Why bad:** High MTTR, frustrated engineers, customer impact

**Fix:** Generate correlation ID at entry point, propagate through all services

```python
# ✅ Good - correlation ID propagation
import uuid
from contextvars import ContextVar

correlation_id_var = ContextVar("correlation_id", default=None)

def handle_request():
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id_var.set(correlation_id)

    # All logs and traces include it automatically
    log.info("processing_request", extra={"correlation_id": correlation_id})
```

## SLI Selection Framework

**Principle:** Measure user experience, not system internals

### Four Golden Signals

| Signal | Definition | Example SLI |
|--------|------------|-------------|
| **Latency** | Request response time | p99 latency < 200ms |
| **Traffic** | Demand on system | Requests per second |
| **Errors** | Failed requests | Error rate < 0.1% |
| **Saturation** | Resource fullness | CPU < 80%, queue depth < 100 |

### RED Method (for services)

- **Rate**: Requests per second
- **Errors**: Error rate (%)
- **Duration**: Response time (p50, p95, p99)

### USE Method (for resources)

- **Utilization**: % time resource busy (CPU %, disk I/O %)
- **Saturation**: Queue depth, wait time
- **Errors**: Error count

**Decision framework:**

| Service Type | Recommended SLIs |
|--------------|------------------|
| **User-facing API** | Availability (%), p95 latency, error rate |
| **Background jobs** | Freshness (time since last run), success rate, processing time |
| **Data pipeline** | Data freshness, completeness (%), processing latency |
| **Storage** | Availability, durability, latency percentiles |

## SLO Definition Guide

**SLO = Target value for SLI**

**Formula:** `SLO = (good events / total events) >= target`

**Example:**
```
SLI: Request success rate
SLO: 99.9% of requests succeed (measured over 30 days)
Error budget: 0.1% = ~43 minutes downtime/month
```

### Error Budget

**Definition:** Amount of unreliability you can tolerate

**Calculation:**
```
Error budget = 1 - SLO target
If SLO = 99.9%, error budget = 0.1%
For 1M requests/month: 1,000 requests can fail
```

**Usage:** Balance reliability vs feature velocity

### Multi-Window Multi-Burn-Rate Alerting

**Problem:** Simple threshold alerts are either too noisy or too slow

**Solution:** Alert based on how fast you're burning error budget

```yaml
# Alert if burning budget 14.4x faster than acceptable (5% in 1 hour)
alert: ErrorBudgetBurnRateHigh
expr: |
  (
    rate(errors[1h]) / rate(requests[1h])
  ) > (14.4 * (1 - 0.999))
annotations:
  summary: "Burning error budget at 14.4x rate"
  runbook: "https://wiki/runbooks/error-budget-burn"
```

## Alert Design Patterns

**Principle:** Alert on symptoms (user impact) not causes (CPU high)

### Symptom-Based Alerting

```python
# ❌ Bad - alert on cause
alert: HighCPU
expr: cpu_usage > 80%

# ✅ Good - alert on symptom
alert: HighLatency
expr: http_request_duration_p99 > 1.0
```

### Alert Severity Levels

| Level | When | Response Time | Example |
|-------|------|---------------|---------|
| **Critical** | User-impacting | Immediate (page) | Error rate >5%, service down |
| **Warning** | Will become critical | Next business day | Error rate >1%, disk 85% full |
| **Info** | Informational | No action needed | Deploy completed, scaling event |

**Rule:** Only page for critical. Everything else goes to dashboard/Slack.

## Cost Optimization Quick Reference

**Observability can cost 5-15% of infrastructure spend. Optimize:**

### Sampling Strategies

```python
# Trace sampling - collect 10% of traces
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

sampler = TraceIdRatioBased(0.1)  # 10% sampling
```

**When to sample:**
- Traces: 1-10% for high-traffic services
- Logs: Sample debug/info logs, keep all errors
- Metrics: Don't sample (they're already aggregated)

### Retention Policies

| Data Type | Recommended Retention | Rationale |
|-----------|----------------------|-----------|
| **Metrics** | 15 days (raw), 13 months (aggregated) | Trend analysis |
| **Logs** | 7-30 days | Debugging, compliance |
| **Traces** | 7 days | Debugging recent issues |

### Cardinality Control

```python
# ❌ Bad - high cardinality
http_requests.labels(
    method=method,
    url=full_url,  # Unbounded!
    user_id=user_id  # Unbounded!
)

# ✅ Good - controlled cardinality
http_requests.labels(
    method=method,
    endpoint=route_pattern,  # /users/:id not /users/12345
    status_code=status
)
```

## Tool Ecosystem Quick Reference

| Category | Open Source | Managed/Commercial |
|----------|-------------|-------------------|
| **Metrics** | Prometheus, VictoriaMetrics | Datadog, New Relic, Grafana Cloud |
| **Logs** | Loki, ELK Stack | Datadog, Splunk, Sumo Logic |
| **Traces** | Jaeger, Zipkin | Datadog, Honeycomb, Lightstep |
| **All-in-One** | Grafana + Loki + Tempo + Mimir | Datadog, New Relic, Dynatrace |
| **Instrumentation** | OpenTelemetry | (vendor SDKs) |

**Recommendation:**
- **Starting out**: Prometheus + Grafana + OpenTelemetry
- **Growing (10-50 services)**: Add Loki (logs) + Jaeger (traces)
- **Scale (50+ services)**: Consider managed (Datadog, Grafana Cloud)

**Why OpenTelemetry:** Vendor-neutral, future-proof, single instrumentation for all signals

## Your First Observability Setup

**Goal:** Metrics + alerting in one week

**Day 1-2: Instrument application**

```python
# Add OpenTelemetry
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader

# Initialize
meter_provider = MeterProvider(
    metric_readers=[PrometheusMetricReader()]
)
metrics.set_meter_provider(meter_provider)

# Instrument HTTP framework (auto-instrumentation)
from opentelemetry.instrumentation.flask import FlaskInstrumentor
FlaskInstrumentor().instrument_app(app)
```

**Day 3-4: Deploy Prometheus + Grafana**

```yaml
# docker-compose.yml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

**Day 5: Define SLIs and SLOs**

```
SLI: HTTP request success rate
SLO: 99.9% of requests succeed (30-day window)
Error budget: 0.1% = 43 minutes downtime/month
```

**Day 6: Create alerts**

```yaml
# prometheus-alerts.yml
groups:
  - name: slo_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Error rate >5% on {{$labels.service}}"
          runbook: "https://wiki/runbooks/high-error-rate"
```

**Day 7: Build dashboard**

**Panels to include:**
- Error rate (%)
- Request rate (req/s)
- p50/p95/p99 latency
- CPU/memory utilization

## Common Mistakes

### ❌ Logging in Production == Debugging in Production
**Fix:** Use structured logging with correlation IDs, not print statements

---

### ❌ Alerting on Predictions, Not Reality
**Fix:** Alert on actual user impact (errors, latency) not predicted issues (disk 70% full)

---

### ❌ Dashboard Sprawl
**Fix:** One main dashboard per service showing SLIs. Delete dashboards unused for 3 months.

---

### ❌ Ignoring Alert Feedback Loop
**Fix:** Track alert precision (% that led to action). Delete alerts with <50% precision.

## Quick Reference

**Getting Started:**
- Start with metrics (Prometheus + OpenTelemetry)
- Add logs when debugging is hard (Loki)
- Add traces when issues span services (Jaeger)

**SLI Selection:**
- User-facing: Availability, latency, error rate
- Background: Freshness, success rate, processing time

**SLO Targets:**
- Start with 99% (achievable)
- Increase to 99.9% only if business requires it
- 99.99% is very expensive (4 nines = 52 min/year downtime)

**Alerting:**
- Critical only = page
- Warning = next business day
- Info = dashboard only

**Cost Control:**
- Sample traces (1-10%)
- Control metric cardinality (no unbounded labels)
- Set retention policies (7-30 days logs, 15 days metrics)

**Tools:**
- Small: Prometheus + Grafana + Loki
- Medium: Add Jaeger
- Large: Consider Datadog, Grafana Cloud

## Bottom Line

**Start with metrics using OpenTelemetry + Prometheus. Define 3-5 SLIs based on user experience. Alert only on symptoms that require immediate action. Add logs and traces when metrics aren't enough.**

Measure what users care about, not what's easy to measure.
