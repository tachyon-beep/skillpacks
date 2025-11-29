---
name: performance-testing-fundamentals
description: Use when starting performance testing, choosing load testing tools, interpreting performance metrics, debugging slow applications, or establishing performance baselines - provides decision frameworks and anti-patterns for load, stress, spike, and soak testing
---

# Performance Testing Fundamentals

## Overview

**Core principle:** Diagnose first, test second. Performance testing without understanding your bottlenecks wastes time.

**Rule:** Define SLAs before testing. You can't judge "good" performance without requirements.

## When NOT to Performance Test

Performance test only AFTER:
- ✅ Defining performance SLAs (latency, throughput, error rate targets)
- ✅ Profiling current bottlenecks (APM, database logs, profiling)
- ✅ Fixing obvious issues (missing indexes, N+1 queries, inefficient algorithms)

**Don't performance test to find problems** - use profiling/APM for that. Performance test to verify fixes and validate capacity.

## Tool Selection Decision Tree

| Your Constraint | Choose | Why |
|----------------|--------|-----|
| CI/CD integration, JavaScript team | **k6** | Modern, code-as-config, easy CI integration |
| Complex scenarios, enterprise, mature ecosystem | **JMeter** | GUI, plugins, every protocol |
| High throughput (10k+ RPS), Scala team | **Gatling** | Built for scale, excellent reports |
| Quick HTTP benchmark, no complex scenarios | **Apache Bench (ab)** or **wrk** | Command-line, no setup |
| Cloud-based, don't want infrastructure | **BlazeMeter**, **Loader.io** | SaaS, pay-per-use |
| Realistic browser testing (JS rendering) | **Playwright** + **k6** | Hybrid: Playwright for UX, k6 for load |

**For most teams:** k6 (modern, scriptable) or JMeter (mature, GUI)

## Test Type Quick Reference

| Test Type | Purpose | Duration | Load Pattern | Use When |
|-----------|---------|----------|--------------|----------|
| **Load Test** | Verify normal operations under expected load | 15-30 min | Steady (ramp to target, sustain) | Baseline validation, regression testing |
| **Stress Test** | Find breaking point | 5-15 min | Increasing (ramp until failure) | Capacity planning, finding limits |
| **Spike Test** | Test sudden traffic surge | 2-5 min | Instant jump (0 → peak) | Black Friday prep, auto-scaling validation |
| **Soak Test** | Find memory leaks, connection pool exhaustion | 2-8 hours | Steady sustained load | Pre-production validation, stability check |

**Start with Load Test** (validates baseline), then Stress/Spike (finds limits), finally Soak (validates stability).

## Anti-Patterns Catalog

### ❌ Premature Load Testing
**Symptom:** "App is slow, let's load test it"

**Why bad:** Load testing reveals "it's slow under load" but not WHY or WHERE

**Fix:** Profile first (APM, database slow query logs, profiler), fix obvious bottlenecks, THEN load test to validate

---

### ❌ Testing Without SLAs
**Symptom:** "My API handles 100 RPS with 200ms average latency. Is that good?"

**Why bad:** Can't judge "good" without requirements. A gaming API needs <50ms; batch processing tolerates 2s.

**Fix:** Define SLAs first:
- Target latency: P95 < 300ms, P99 < 500ms
- Target throughput: 500 RPS at peak
- Max error rate: < 0.1%

---

### ❌ Unrealistic SLAs
**Symptom:** "Our database-backed CRUD API with complex joins must have P95 < 10ms"

**Why bad:** Sets impossible targets. Database round-trip alone is often 5-20ms. Forces wasted optimization or architectural rewrites.

**Fix:** Compare against Performance Benchmarks table (see below). If target is 10x better than benchmark, profile current performance first, then negotiate realistic SLA based on what's achievable vs cost of optimization.

---

### ❌ Vanity Metrics
**Symptom:** Reporting only average response time

**Why bad:** Average hides tail latency. 99% of requests at 100ms + 1% at 10s = "average 200ms" looks fine, but users experience 10s delays.

**Fix:** Always report percentiles:
- P50 (median) - typical user experience
- P95 - most users
- P99 - worst-case for significant minority
- Max - outliers

---

### ❌ Load Testing in Production First
**Symptom:** "Let's test capacity by running load tests against production"

**Why bad:** Risks outages, contaminates real metrics, can trigger alerts/costs

**Fix:** Test in staging environment that mirrors production (same DB size, network latency, resource limits)

---

### ❌ Single-User "Load" Tests
**Symptom:** Running one user hitting the API as fast as possible

**Why bad:** Doesn't simulate realistic concurrency, misses resource contention (database connections, thread pools)

**Fix:** Simulate realistic concurrent users with realistic think time between requests

## Metrics Glossary

| Metric | Definition | Good Threshold (typical web API) |
|--------|------------|----------------------------------|
| **RPS** (Requests/Second) | Throughput - how many requests processed | Varies by app; know your peak |
| **Latency** | Time from request to response | P95 < 300ms, P99 < 500ms |
| **P50 (Median)** | 50% of requests faster than this | P50 < 100ms |
| **P95** | 95% of requests faster than this | P95 < 300ms |
| **P99** | 99% of requests faster than this | P99 < 500ms |
| **Error Rate** | % of 4xx/5xx responses | < 0.1% |
| **Throughput** | Data transferred per second (MB/s) | Depends on payload size |
| **Concurrent Users** | Active users at same time | Calculate from traffic patterns |

**Focus on P95/P99, not average.** Tail latency kills user experience.

## Diagnostic-First Workflow

Before load testing slow applications, follow this workflow:

**Step 1: Measure Current State**
- Install APM (DataDog, New Relic, Grafana) or logging
- Identify slowest 10 endpoints/operations
- Check database slow query logs

**Step 2: Common Quick Wins** (90% of performance issues)
- Missing database indexes
- N+1 query problem
- Unoptimized images/assets
- Missing caching (Redis, CDN)
- Synchronous operations that should be async
- Inefficient serialization (JSON parsing bottlenecks)

**Step 3: Profile Specific Bottleneck**
- Use profiler to see CPU/memory hotspots
- Trace requests to find where time is spent (DB? external API? computation?)
- Check for resource limits (max connections, thread pool exhaustion)

**Step 4: Fix and Measure**
- Apply fix (add index, cache layer, async processing)
- Measure improvement in production
- Document before/after metrics

**Step 5: THEN Load Test** (if needed)
- Validate fixes handle expected load
- Find new capacity limits
- Establish regression baseline

**Anti-pattern to avoid:** Skipping to Step 5 without Steps 1-4.

## Performance Benchmarks (Reference)

What "good" looks like by application type:

| Application Type | Typical P95 Latency | Typical Throughput | Notes |
|------------------|---------------------|-------------------|-------|
| **REST API (CRUD)** | < 200ms | 500-2000 RPS | Database-backed, simple queries |
| **Search API** | < 500ms | 100-500 RPS | Complex queries, ranking algorithms |
| **Payment Gateway** | < 1s | 50-200 RPS | External service calls, strict consistency |
| **Real-time Gaming** | < 50ms | 1000-10000 RPS | Low latency critical |
| **Batch Processing** | 2-10s/job | 10-100 jobs/min | Throughput > latency |
| **Static CDN** | < 100ms | 10000+ RPS | Edge-cached, minimal computation |

**Use as rough guide, not absolute targets.** Your SLAs depend on user needs.

## Results Interpretation Framework

After running a load test:

**Pass Criteria:**
- ✅ All requests meet latency SLA (e.g., P95 < 300ms)
- ✅ Error rate under threshold (< 0.1%)
- ✅ No resource exhaustion (CPU < 80%, memory stable, no connection pool saturation)
- ✅ Sustained load for test duration without degradation

**Fail Criteria:**
- ❌ Latency exceeds SLA
- ❌ Error rate spikes
- ❌ Gradual degradation over time (memory leak, connection leak)
- ❌ Resource exhaustion (CPU pegged, OOM errors)

**Next Steps:**
- **If passing:** Establish this as regression baseline, run periodically in CI
- **If failing:** Profile to find bottleneck, optimize, re-test
- **If borderline:** Test at higher load (stress test) to find safety margin

## Common Mistakes

### ❌ Not Ramping Load Gradually
**Symptom:** Instant 0 → 1000 users, everything fails

**Fix:** Ramp over 2-5 minutes to let auto-scaling/caching warm up (except spike tests, where instant jump is the point)

---

### ❌ Testing With Empty Database
**Symptom:** Tests pass with 100 records, fail with 1M records in production

**Fix:** Seed staging database with production-scale data

---

### ❌ Ignoring External Dependencies
**Symptom:** Your API is fast, but third-party payment gateway times out under load

**Fix:** Include external service latency in SLAs, or mock them for isolated API testing

## Quick Reference

**Getting Started Checklist:**
1. Define SLAs (latency P95/P99, throughput, error rate)
2. Choose tool (k6 or JMeter for most cases)
3. Start with Load Test (baseline validation)
4. Run Stress Test (find capacity limits)
5. Establish regression baseline
6. Run in CI on major changes

**When Debugging Slow App:**
1. Profile first (APM, database logs)
2. Fix obvious issues (indexes, N+1, caching)
3. Measure improvement
4. THEN load test to validate

**Interpreting Results:**
- Report P95/P99, not just average
- Compare against SLAs
- Check for resource exhaustion
- Look for degradation over time (soak tests)

## Bottom Line

**Performance testing validates capacity and catches regressions.**

**Profiling finds bottlenecks.**

Don't confuse the two - diagnose first, test second.
