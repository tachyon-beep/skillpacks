---
name: performance-testing-foundations
description: Use when validating system performance under load - covers load testing, stress testing, endurance testing, baseline establishment, performance regression detection, and bottleneck identification
---

# Performance Testing Foundations

## Overview

Performance testing validates that your system meets speed, scalability, and stability requirements under expected and peak loads. Without performance testing, you discover scalability issues in production - when it's too late.

**Core Principle**: Establish baselines first, then test for regressions. Use percentiles (p95, p99), not averages. Load increases gradually to find breaking points.

**Ordis Identity**: Performance testing is proactive defense against production overload - controlled stress to identify weaknesses before users do.

## When to Use

**Use this skill when**:
- Validating system can handle expected load
- Preventing performance regressions
- Finding capacity limits and bottlenecks
- Load testing new features or infrastructure changes
- Establishing SLOs for latency/throughput
- Debugging slow requests or high latency

**Don't use for**:
- Unit test performance (use micro-benchmarks instead)
- Production monitoring (use observability skills)
- One-time performance investigation (unless establishing baseline)

## Types of Performance Testing

### Load Testing

**What**: Test system under expected normal and peak load.

**Goal**: Verify system handles typical traffic patterns.

**Example**: "Can API handle 1000 requests/second during business hours?"

**Pattern**:
```
Load (requests/sec)
    ^
    |     ┌────────────┐  Sustained peak load
1000├─────┤            │
    |    /              \
 500├───/                \───
    |  /                    \
    └──────────────────────────> Time
      Ramp-up    Sustain    Ramp-down
```

### Stress Testing

**What**: Test system beyond expected load until it breaks.

**Goal**: Find breaking point and failure mode.

**Example**: "At what point does API start failing? How does it fail?"

**Pattern**:
```
Load (requests/sec)
    ^
    |                  ┌─ Breaking point (errors, timeouts)
3000├──────────────────┤
    |                /
2000├──────────────/
    |            /
1000├──────────/
    |        /
    └────────────────────> Time
         Incremental increase
```

### Spike Testing

**What**: Sudden, dramatic load increase.

**Goal**: Verify system handles traffic spikes (product launches, marketing campaigns).

**Example**: "What happens when traffic jumps 10x in 1 minute?"

**Pattern**:
```
Load (requests/sec)
    ^
    |     ┌─┐
5000├─────┤ │  Sudden spike
    |     │ │
    |     │ │
 500├─────┘ └─────
    |
    └──────────────> Time
```

### Endurance Testing (Soak Testing)

**What**: Sustained load over extended period (hours, days).

**Goal**: Find memory leaks, resource exhaustion, degradation over time.

**Example**: "Does API remain stable under 500 req/s for 24 hours?"

**Pattern**:
```
Load (requests/sec)
    ^
    |  ┌────────────────────────────────┐
 500├──┤                                │
    |  │    Hours or days               │
    |  └────────────────────────────────┘
    └──────────────────────────────────────> Time
                (24-72 hours)
```

### Scalability Testing

**What**: Test horizontal/vertical scaling effectiveness.

**Goal**: Verify adding resources increases capacity linearly.

**Example**: "Does doubling instances double throughput?"

## Metrics That Matter

### ❌ Wrong Metrics

**Average response time** - Hides outliers, masks problems.

```
Request latencies: [10ms, 12ms, 11ms, 10ms, 3000ms]
Average: 608ms ← Looks bad
Reality: 80% of requests are <15ms, 1 slow outlier
```

**Maximum response time** - Single outlier, not representative.

### ✅ Right Metrics

**Percentiles (p50, p95, p99)**

- **p50 (median)**: 50% of requests faster than this
- **p95**: 95% of requests faster than this (1 in 20 requests slower)
- **p99**: 99% of requests faster than this (1 in 100 requests slower)

```
1000 requests:
p50: 45ms   ← Half of users see <45ms
p95: 120ms  ← 95% of users see <120ms
p99: 450ms  ← 99% of users see <450ms
max: 3000ms ← Outlier
```

**Why percentiles**: They reveal tail latency (what your slowest users experience).

**Throughput** - Requests per second successfully processed.

```
Target: 1000 req/s
Actual: 950 req/s with 0% errors ✓
Actual: 1200 req/s with 25% errors ✗ (not handling load)
```

**Error rate** - Percentage of failed requests.

```
Error rate:
0-1%: ✓ Good
1-5%: ⚠ Warning
>5%:  ✗ System degraded
```

**Resource utilization** - CPU, memory, disk I/O, network.

```
CPU: 40% ← Healthy (room to scale)
CPU: 85% ← Nearing capacity (bottleneck)
CPU: 95% ← At capacity (errors likely)
```

## Establishing Baselines

**Baseline**: Performance snapshot before changes. Used to detect regressions.

### Step 1: Baseline Test

**Before** making changes:

```bash
# Run baseline load test
k6 run --vus 50 --duration 5m baseline-test.js

# Sample output
✓ http_req_duration..............: avg=45ms  p95=120ms p99=180ms
✓ http_req_failed................: 0.2%
✓ http_reqs......................: 15000 (50/s)
```

**Save results**: Store metrics in file or monitoring system.

```json
{
  "baseline": "2025-01-15",
  "version": "v2.3.0",
  "p50": 30,
  "p95": 120,
  "p99": 180,
  "throughput": 50,
  "error_rate": 0.002
}
```

### Step 2: Make Changes

Deploy new feature, infrastructure change, code refactor.

### Step 3: Compare to Baseline

```bash
# Run same test after changes
k6 run --vus 50 --duration 5m baseline-test.js

# Compare results
✓ http_req_duration..............: avg=48ms  p95=130ms p99=190ms
                                    ↑ +6%     ↑ +8%     ↑ +5%
```

**Decision criteria**:
- **< 10% degradation**: Accept (within noise)
- **10-25% degradation**: Investigate
- **> 25% degradation**: Reject, find root cause

## Load Testing Tools

### k6 (Recommended for APIs)

**Advantages**: JavaScript-based, scriptable, good CLI output, cloud integration.

```javascript
// k6 load test script
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp-up to 100 VUs
    { duration: '5m', target: 100 },  // Sustain 100 VUs
    { duration: '2m', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p95<200'],   // 95% of requests < 200ms
    http_req_failed: ['rate<0.01'],   // Error rate < 1%
  },
};

export default function () {
  const res = http.get('https://api.example.com/products');

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time OK': (r) => r.timings.duration < 500,
  });

  sleep(1); // 1 second between requests per VU
}
```

**Run**:
```bash
k6 run load-test.js

# With results output
k6 run --out json=results.json load-test.js
```

### Locust (Recommended for Complex Scenarios)

**Advantages**: Python-based, distributed load testing, web UI for monitoring.

```python
# locustfile.py
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3s between requests

    @task(3)  # Weight: 3x more likely than other tasks
    def view_products(self):
        self.client.get("/products")

    @task(1)
    def view_product(self):
        product_id = random.randint(1, 100)
        self.client.get(f"/products/{product_id}")

    @task(1)
    def add_to_cart(self):
        self.client.post("/cart", json={
            "product_id": random.randint(1, 100),
            "quantity": 1
        })
```

**Run**:
```bash
# Web UI (localhost:8089)
locust -f locustfile.py --host=https://api.example.com

# Headless mode
locust -f locustfile.py --host=https://api.example.com \
  --users 100 --spawn-rate 10 --run-time 10m --headless
```

### JMeter (Enterprise Standard)

**Advantages**: GUI, wide protocol support, extensive plugins, enterprise adoption.

**Disadvantages**: XML-based, heavy, harder to version control.

**Use when**: Enterprise environment requires JMeter, testing protocols beyond HTTP.

### Artillery (Simpler Alternative)

**Advantages**: YAML-based, simple syntax, good for quick tests.

```yaml
# artillery-test.yml
config:
  target: 'https://api.example.com'
  phases:
    - duration: 60
      arrivalRate: 10  # 10 users per second

scenarios:
  - name: "Browse products"
    flow:
      - get:
          url: "/products"
      - think: 2
      - get:
          url: "/products/{{ $randomNumber(1, 100) }}"
```

**Run**:
```bash
artillery run artillery-test.yml
```

**Recommendation**: Use k6 for APIs (best balance of power/simplicity), Locust for complex scenarios or Python teams.

## Load Profiles

### Constant Load

**Pattern**: Fixed number of VUs (virtual users) for duration.

```javascript
// k6 constant load
export const options = {
  vus: 50,
  duration: '10m',
};
```

**Use when**: Testing sustained throughput, establishing baseline.

### Ramp-Up Load

**Pattern**: Gradually increase VUs to target.

```javascript
// k6 ramp-up
export const options = {
  stages: [
    { duration: '5m', target: 100 },  // Ramp from 0 to 100
    { duration: '10m', target: 100 }, // Sustain 100
    { duration: '5m', target: 0 },    // Ramp down
  ],
};
```

**Use when**: Finding breaking point, avoiding coordinated omission (see below).

### Step Load

**Pattern**: Step-wise increase to find capacity limits.

```javascript
// k6 step load
export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Step 1
    { duration: '2m', target: 100 },  // Step 2
    { duration: '2m', target: 150 },  // Step 3
    { duration: '2m', target: 200 },  // Step 4
  ],
};
```

**Use when**: Finding capacity limits systematically.

## Coordinated Omission

**Critical Problem**: Load testing tools can hide latency if not configured correctly.

**Scenario**:
```
Target: 100 requests/second
Reality: System slows down, handles 50 requests/second
```

**Wrong approach** (coordinated omission):
```
Tool waits for response before sending next request
→ Automatically slows down to match system capacity
→ Reported latency looks good (only successful requests measured)
→ Hides that system is dropping/delaying requests
```

**Right approach** (open model):
```
Tool sends requests at fixed rate regardless of responses
→ Requests queue up if system is slow
→ Tool measures actual latency including queuing
→ Reveals true user experience
```

**k6 example**:
```javascript
// ❌ WRONG: Closed model (coordinated omission)
export const options = {
  vus: 100,  // 100 concurrent users
};

export default function () {
  http.get('https://api.example.com/products');
  sleep(1);  // Wait 1s, then next request
  // If request takes 5s, actual rate is 1 req/6s per VU
}

// ✅ RIGHT: Open model (fixed arrival rate)
export const options = {
  scenarios: {
    constant_request_rate: {
      executor: 'constant-arrival-rate',
      rate: 100,      // 100 requests/second
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 50,
      maxVUs: 200,    // Scale VUs if needed
    },
  },
};

export default function () {
  http.get('https://api.example.com/products');
  // No sleep - requests sent at fixed rate
}
```

**Takeaway**: Use arrival rate-based load (open model), not VU-based (closed model).

## Identifying Bottlenecks

**When performance degrades, find the bottleneck:**

### CPU-Bound

**Symptoms**: CPU at 90-100%, response times increase linearly with load.

**Diagnosis**:
```bash
# Check CPU usage during load test
top
htop

# Profile application (example: Node.js)
node --prof app.js  # Run under load
node --prof-process isolate-*.log > profile.txt
```

**Solutions**:
- Optimize hot code paths (profiling shows where time is spent)
- Add caching for expensive computations
- Scale horizontally (more instances)
- Scale vertically (more CPU cores)

### Memory-Bound

**Symptoms**: High memory usage, GC pressure, OOM errors.

**Diagnosis**:
```bash
# Check memory usage
free -m
vmstat 1

# Heap snapshot (Node.js)
node --inspect app.js
# Take heap snapshot during load test in Chrome DevTools
```

**Solutions**:
- Fix memory leaks
- Reduce memory footprint (e.g., pagination instead of loading all data)
- Increase memory limit
- Scale horizontally

### Database-Bound

**Symptoms**: Database CPU/connections maxed, slow queries.

**Diagnosis**:
```sql
-- PostgreSQL: Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check connection pool usage
SELECT count(*) FROM pg_stat_activity;
```

**Solutions**:
- Add indexes for slow queries
- Optimize queries (EXPLAIN ANALYZE)
- Increase connection pool size
- Add read replicas
- Add caching layer (Redis)

### Network-Bound

**Symptoms**: High network latency, bandwidth saturation.

**Diagnosis**:
```bash
# Check network throughput
iftop
nethogs

# Test network latency
ping api.example.com
traceroute api.example.com
```

**Solutions**:
- Use CDN for static assets
- Enable compression (gzip, Brotli)
- Reduce payload sizes
- Use HTTP/2 or HTTP/3
- Scale horizontally across regions

### Third-Party Dependency

**Symptoms**: Response times spike when calling external APIs.

**Diagnosis**: Check traces (distributed tracing with OpenTelemetry, Jaeger).

**Solutions**:
- Add timeout/circuit breaker
- Add caching layer
- Use async processing (queue)
- Negotiate SLAs with vendor

## Performance Budgets

**Performance budget**: Maximum acceptable latency/throughput for each endpoint.

### Define Budgets

```yaml
# performance-budget.yml
budgets:
  /api/products:
    p95_latency_ms: 200
    p99_latency_ms: 500
    throughput_rps: 100
    error_rate: 0.01

  /api/checkout:
    p95_latency_ms: 500  # Slower is acceptable (complex operation)
    p99_latency_ms: 1000
    throughput_rps: 50
    error_rate: 0.001  # Stricter (revenue-critical)
```

### Enforce in CI

```javascript
// k6 test with thresholds (budgets)
export const options = {
  thresholds: {
    'http_req_duration{endpoint:/api/products}': ['p95<200', 'p99<500'],
    'http_req_failed{endpoint:/api/products}': ['rate<0.01'],
  },
};

// Test fails if thresholds violated
```

**CI integration**:
```bash
# Run performance test in CI
k6 run --out json=results.json load-test.js

# Exit code 0 if pass, 1 if fail (thresholds violated)
echo $?  # 0 = pass, 1 = fail
```

**Benefit**: Catch performance regressions before merge.

## Quick Reference

| Testing Type | Goal | Duration | Pattern |
|--------------|------|----------|---------|
| **Load** | Validate expected capacity | 5-30 min | Ramp to target, sustain |
| **Stress** | Find breaking point | 10-60 min | Incremental increase until failure |
| **Spike** | Handle sudden traffic | 5-15 min | Sudden jump, brief sustain |
| **Endurance** | Find memory leaks, degradation | 6-72 hours | Constant load, long duration |
| **Scalability** | Verify horizontal scaling | 30-60 min | Compare performance: 1 vs 2 vs 4 instances |

| Metric | Use | Don't Use |
|--------|-----|-----------|
| **p95, p99 latency** | ✅ Percentiles reveal tail latency | ❌ Average (hides outliers) |
| **Error rate** | ✅ Percentage of failed requests | ❌ Error count (depends on throughput) |
| **Throughput (req/s)** | ✅ Successful requests per second | ❌ Total requests (includes errors) |
| **Resource utilization** | ✅ CPU, memory, disk, network usage | ❌ Single metric in isolation |

## Common Mistakes

### ❌ No Baseline

**Wrong**: Run load test, see numbers, "looks good?"
**Right**: Establish baseline, compare changes to baseline

**Why**: Without baseline, you can't detect regressions.

### ❌ Using Averages

**Wrong**: "Average response time is 50ms, we're good"
**Right**: "p95 is 200ms, p99 is 500ms"

**Why**: Averages hide tail latency (what slow users experience).

### ❌ Testing in Production First

**Wrong**: "Let's load test in production to see what happens"
**Right**: Test in staging/pre-prod first, then production (with monitoring)

**Why**: Load testing can cause outages if system can't handle load.

### ❌ Ignoring Coordinated Omission

**Wrong**: Use closed-loop model (VUs with sleep)
**Right**: Use open-loop model (fixed arrival rate)

**Why**: Closed-loop hides latency when system slows down.

### ❌ Testing One Component in Isolation

**Wrong**: Load test API without database, cache, third-party services
**Right**: Test full system (or use realistic mocks)

**Why**: Bottlenecks often appear at integration points.

### ❌ Not Monitoring During Test

**Wrong**: Run load test, only look at k6 output
**Right**: Monitor CPU, memory, database, logs during test

**Why**: Need to identify bottleneck location, not just symptoms.

## Real-World Impact

**Before Performance Testing**:
- Launched feature to 100% of users
- API latency spiked from 50ms to 5000ms
- Database connections maxed out
- Rolled back in emergency
- 2-hour outage

**After Performance Testing**:
- Baseline: p95 = 50ms at 100 req/s
- Load tested new feature: p95 = 250ms at 100 req/s (5x degradation)
- Found N+1 query in new code path
- Fixed before launch
- No production incident

## Summary

**Performance testing is controlled stress to find weaknesses before production:**

1. **Establish baselines** before changes (save metrics)
2. **Use percentiles (p95, p99)**, not averages
3. **Test types**: Load (expected), stress (breaking point), spike (sudden), endurance (memory leaks)
4. **Avoid coordinated omission** - use fixed arrival rate, not VU-based
5. **Identify bottlenecks** - CPU, memory, database, network, or third-party
6. **Enforce budgets** - p95/p99 thresholds in CI
7. **Monitor during tests** - find bottleneck location

**Ordis Principle**: Test the bulwark under fire before the real siege arrives. Performance testing is proactive defense.
