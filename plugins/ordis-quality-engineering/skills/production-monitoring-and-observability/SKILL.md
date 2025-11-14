---
name: production-monitoring-and-observability
description: Use when establishing production quality signals through metrics, logging, tracing, and alerting - provides SLI/SLO/SLA definitions, alert fatigue prevention, distributed tracing, and observability-driven development
---

# Production Monitoring and Observability

## Overview

Monitoring tells you when something is wrong. Observability tells you why. Without observability, you're blind in production - reacting to user complaints instead of proactive signals. With observability, you detect issues before users do and diagnose root causes quickly.

**Core Principle**: Define SLOs (service level objectives) first, alert on symptoms (not causes), use three pillars (metrics, logs, traces). Prevent alert fatigue through aggregation and intelligent thresholds.

**Ordis Identity**: Observability is the Gestalt for distributed systems - collective consciousness linking all components, revealing system health and enabling coordinated response.

## When to Use

**Use this skill when**:
- Setting up monitoring for new services
- Production incidents lack visibility
- Alert fatigue (too many alerts, team ignores them)
- Unclear which metrics matter
- Distributed tracing needed for multi-service debugging
- Defining SLOs and error budgets

**Don't use for**:
- Local development debugging (use debugger)
- Load testing metrics (use performance-testing-foundations)
- Pre-production testing (use test-automation-strategy)

## Three Pillars of Observability

### 1. Metrics (What)

**Quantitative measurements over time**: request rate, error rate, latency, CPU usage.

**Types**:
- **Counters**: Monotonically increasing (total requests, errors)
- **Gauges**: Current value (CPU %, memory usage, queue depth)
- **Histograms**: Distribution (latency percentiles)

**Example metrics**:
```
http_requests_total{status="200"} 15000
http_requests_total{status="500"} 25
http_request_duration_p95 125ms
cpu_usage_percent 45
```

### 2. Logs (Why)

**Discrete events with context**: errors, warnings, user actions.

**Structured logging**:
```json
{
  "timestamp": "2025-01-15T14:30:00Z",
  "level": "error",
  "message": "Payment processing failed",
  "user_id": "usr_12345",
  "order_id": "ord_67890",
  "error": "Insufficient funds",
  "trace_id": "abc123def456"
}
```

**Anti-pattern (unstructured)**:
```
2025-01-15 14:30:00 ERROR: Payment failed for user usr_12345
```

### 3. Traces (Where)

**Request journey across services**: shows latency breakdown, bottlenecks.

```
Trace ID: abc123def456

API Gateway ────[50ms]────> Order Service ────[200ms]────> Payment Service
                                    │
                                    └──[30ms]──> Inventory Service
```

**Shows**:
- Total latency: 280ms
- Bottleneck: Payment Service (200ms)
- Parallelization opportunity: Inventory check

## SLI, SLO, SLA

### SLI (Service Level Indicator)

**What**: Quantitative measure of service quality.

**Examples**:
- Request success rate
- p95 latency
- Availability (uptime %)
- Throughput (requests/second)

```
SLI: 99.5% of requests return 2xx status
SLI: p95 latency < 200ms
SLI: System uptime 99.9%
```

### SLO (Service Level Objective)

**What**: Target value for an SLI. Internal goal.

**Example**:
```
SLO: 99.9% availability (43 minutes downtime/month allowed)
SLO: p95 latency < 200ms for 99% of 5-minute windows
SLO: Error rate < 0.1%
```

**Error budget**: Allowable deviation from SLO.

```
SLO: 99.9% availability
Error budget: 0.1% = 43 minutes downtime per month

If error budget exhausted: Stop feature development, focus on reliability
```

### SLA (Service Level Agreement)

**What**: Contract with customers, includes consequences.

**Example**:
```
SLA: 99.5% uptime or customers get 10% refund
```

**Relationship**: SLA ≤ SLO ≤ actual performance

```
99.95% actual performance
    ↓
99.9% SLO (internal goal)
    ↓
99.5% SLA (customer promise)
```

## Alert Design

### Alert on Symptoms, Not Causes

**❌ Cause-based alert**:
```
Alert: Database CPU > 80%
```

**Problem**: High CPU might not affect users. Alert fatigue.

**✅ Symptom-based alert**:
```
Alert: Error rate > 1% for 5 minutes
  OR
Alert: p95 latency > 500ms for 5 minutes
```

**Benefit**: Alerts when users affected, not when infrastructure is busy.

### Multi-Window Multi-Burn-Rate Alerts

**Concept**: Alert on multiple timescales to catch fast and slow burns.

```
# Fast burn (exhausts error budget in hours)
Alert: Error rate > 5% for 5 minutes
  → Page on-call immediately

# Slow burn (exhausts error budget in days)
Alert: Error rate > 0.5% for 1 hour
  → Create ticket, investigate during business hours
```

**Benefits**:
- Catch severe incidents fast (page on-call)
- Catch gradual degradation early (ticket)

### Actionable Alerts

**Every alert must have**:
1. **Clear condition**: What triggered the alert
2. **Impact**: What users experience
3. **Runbook link**: How to investigate/fix
4. **Recent changes**: What deployed recently

**Example alert**:
```
🚨 High Error Rate (P1)

Condition: 5.2% of requests failing (threshold: 1%)
Impact: Users seeing "Service Unavailable" errors
Duration: 8 minutes
Recent changes: API v2.3.5 deployed 15 minutes ago

Runbook: https://wiki.example.com/runbooks/high-error-rate
Grafana: https://grafana.example.com/dashboard/errors
Logs: https://logs.example.com?trace_id=abc123
```

## Alert Fatigue Prevention

**Alert fatigue**: Too many alerts → team ignores them → miss real incidents.

### Techniques

**1. Aggregation**:
```
❌ Bad: 100 alerts for 100 failing instances
✅ Good: 1 alert "API error rate > 5%"
```

**2. Suppression during incidents**:
```
# If high error rate alert firing, suppress related alerts
Alert: High error rate
  → Suppress: High latency, Low throughput (symptoms of same issue)
```

**3. Escalation policies**:
```
Alert triggered
  ↓
5 minutes: Notify on-call (PagerDuty)
  ↓ (if not acknowledged)
15 minutes: Escalate to manager
  ↓ (if not acknowledged)
30 minutes: Escalate to VP Engineering
```

**4. Alert tuning**:
```
Alert fires too often (false positives)
  → Increase threshold or lengthen window

Alert fires too late (misses incidents)
  → Decrease threshold or shorten window
```

## Metrics Frameworks

### RED Method (Services)

**For every service**, track:

- **R**ate: Requests per second
- **E**rrors: Error rate (%)
- **D**uration: Latency (p50, p95, p99)

```
http_requests_per_second{service="api"} 150
http_error_rate{service="api"} 0.5%
http_duration_p95{service="api"} 200ms
```

### USE Method (Resources)

**For every resource** (CPU, disk, network), track:

- **U**tilization: % time resource busy
- **S**aturation: Backlog/queue depth
- **E**rrors: Error count

```
cpu_utilization{host="api-1"} 65%
disk_queue_depth{host="api-1"} 5
network_errors{host="api-1"} 0
```

### Four Golden Signals (Google SRE)

1. **Latency**: Time to serve requests
2. **Traffic**: Demand (requests/second)
3. **Errors**: Failed requests
4. **Saturation**: How "full" the service is

## Distributed Tracing

**Problem**: Requests span multiple services - hard to debug latency.

**Solution**: Trace requests with unique ID across all services.

### Implementation (OpenTelemetry)

```javascript
// API Gateway
const trace = require('@opentelemetry/api');

app.get('/checkout', async (req, res) => {
  const span = trace.getTracer('api-gateway').startSpan('checkout');
  const traceId = span.spanContext().traceId;

  // Pass trace ID to downstream services
  const order = await orderService.createOrder({
    userId: req.user.id,
    traceId: traceId
  });

  const payment = await paymentService.charge({
    orderId: order.id,
    traceId: traceId
  });

  span.end();
  res.json({ orderId: order.id });
});
```

```javascript
// Order Service
orderService.createOrder = async ({ userId, traceId }) => {
  const span = trace.getTracer('order-service')
    .startSpan('create-order', { traceId });

  // Create order
  const order = await db.orders.create({ userId });

  // Call inventory service
  await inventoryService.reserve({ orderId: order.id, traceId });

  span.end();
  return order;
};
```

**Trace visualization**:
```
Trace: abc123def456 (total: 350ms)

API Gateway (50ms)
  └─> Order Service (250ms)
        ├─> Inventory Service (30ms)
        └─> Payment Service (200ms) ← BOTTLENECK
```

**Tools**: Jaeger, Zipkin, Datadog APM, New Relic

## Structured Logging

**Anti-pattern (unstructured)**:
```javascript
console.log(`User ${userId} checkout failed: ${error}`);
```

**Problem**: Hard to query, parse, aggregate.

**Pattern (structured)**:
```javascript
logger.error('Checkout failed', {
  user_id: userId,
  order_id: orderId,
  error_message: error.message,
  error_code: error.code,
  trace_id: req.traceId,
  ip_address: req.ip,
  user_agent: req.headers['user-agent']
});
```

**Output (JSON)**:
```json
{
  "timestamp": "2025-01-15T14:30:00Z",
  "level": "error",
  "message": "Checkout failed",
  "user_id": "usr_12345",
  "order_id": "ord_67890",
  "error_message": "Insufficient funds",
  "error_code": "INSUFFICIENT_FUNDS",
  "trace_id": "abc123",
  "ip_address": "203.0.113.42",
  "user_agent": "Mozilla/5.0..."
}
```

**Querying (SQL-like)**:
```sql
SELECT user_id, COUNT(*) as error_count
FROM logs
WHERE level = 'error'
  AND error_code = 'INSUFFICIENT_FUNDS'
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY user_id
ORDER BY error_count DESC
LIMIT 10
```

## Dashboards

### Critical Metrics Dashboard

**For on-call**: Quick system health overview

```
┌─────────────────────────────────────────────┐
│ API Health                                  │
│ ├─ Request Rate: 150/s                      │
│ ├─ Error Rate: 0.3% ✓                       │
│ ├─ p95 Latency: 185ms ✓                     │
│ └─ p99 Latency: 420ms ⚠                     │
│                                             │
│ Database                                    │
│ ├─ Connection Pool: 45/100 ✓                │
│ ├─ Query p95: 50ms ✓                        │
│ └─ Replication Lag: 2s ✓                    │
│                                             │
│ Infrastructure                              │
│ ├─ CPU: 45% ✓                               │
│ ├─ Memory: 62% ✓                            │
│ └─ Disk: 78% ⚠                              │
└─────────────────────────────────────────────┘
```

### Service-Specific Dashboard

**For service owners**: Deep dive into service health

```
Service: Order API

RED Metrics (last 1 hour)
├─ Rate: [graph showing requests/s over time]
├─ Errors: [graph showing error rate over time]
└─ Duration: [graph showing p50/p95/p99 latency]

Top Endpoints by Latency
1. POST /checkout - p95: 450ms
2. GET /orders/:id - p95: 120ms
3. POST /orders - p95: 80ms

Recent Errors (last 15 min)
- PaymentDeclinedError: 25 occurrences
- TimeoutError: 5 occurrences
- ValidationError: 3 occurrences
```

## Quick Reference

| Pillar | Use For | Tools |
|--------|---------|-------|
| **Metrics** | Time-series data, trends, dashboards | Prometheus, Datadog, CloudWatch |
| **Logs** | Discrete events, debugging, search | ELK, Loki, Splunk |
| **Traces** | Request flow, latency breakdown | Jaeger, Zipkin, Datadog APM |

| Alert Type | When to Fire | Escalation |
|------------|--------------|------------|
| **P1 (Critical)** | Error rate > 5% for 5 min | Page on-call immediately |
| **P2 (High)** | Error rate > 1% for 15 min | Notify on-call (not page) |
| **P3 (Medium)** | Error rate > 0.5% for 1 hour | Create ticket |

## Common Mistakes

### ❌ Alerting on Everything

**Wrong**: Alert on every error, every slow query, every spike
**Right**: Alert on SLO violations (symptoms users experience)

**Why**: Too many alerts = alert fatigue = ignored alerts.

### ❌ Unstructured Logging

**Wrong**: `console.log("Error: " + error)`
**Right**: `logger.error("Checkout failed", { userId, orderId, error })`

**Why**: Structured logs are queryable, aggregatable.

### ❌ No Runbooks

**Wrong**: Alert fires, on-call doesn't know what to do
**Right**: Every alert links to runbook with investigation steps

**Why**: Faster incident response.

### ❌ Monitoring Without SLOs

**Wrong**: "Let's monitor everything"
**Right**: "Define SLOs first, then monitor SLIs"

**Why**: SLOs define what matters to users.

## Summary

**Observability provides visibility into production system health:**

1. **Three pillars**: Metrics (what), Logs (why), Traces (where)
2. **Define SLOs** (service level objectives) first
3. **Alert on symptoms** (user impact), not causes (infrastructure)
4. **Prevent alert fatigue** (aggregation, suppression, tuning)
5. **Structured logging** (queryable, aggregatable)
6. **Distributed tracing** (request flow across services)
7. **Actionable alerts** (runbook, recent changes, impact)

**Ordis Principle**: Observability is collective consciousness - linking all components to reveal system health and enable coordinated defense.
