---
name: chaos-engineering-principles
description: Use when starting chaos engineering, designing fault injection experiments, choosing chaos tools, testing system resilience, or recovering from chaos incidents - provides hypothesis-driven testing, blast radius control, and anti-patterns for safe chaos
---

# Chaos Engineering Principles

## Overview

**Core principle:** Chaos engineering validates resilience through controlled experiments, not random destruction.

**Rule:** Start in staging, with monitoring, with rollback, with small blast radius. No exceptions.

## When NOT to Do Chaos

Don't run chaos experiments if ANY of these are missing:
- ❌ No comprehensive monitoring (APM, metrics, logs, alerts)
- ❌ No automated rollback capability
- ❌ No baseline metrics documented
- ❌ No incident response team available
- ❌ System already unstable (fix stability first)
- ❌ No staging environment to practice

**Fix these prerequisites BEFORE chaos testing.**

## Tool Selection Decision Tree

| Your Constraint | Choose | Why |
|----------------|--------|-----|
| Kubernetes-native, CNCF preference | **LitmusChaos** | Cloud-native, operator-based, excellent K8s integration |
| Kubernetes-focused, visualization needs | **Chaos Mesh** | Fine-grained control, dashboards, low overhead |
| Want managed service, quick start | **Gremlin** | Commercial, guided experiments, built-in best practices |
| Vendor-neutral, maximum flexibility | **Chaos Toolkit** | Open source, plugin ecosystem, any infrastructure |
| AWS-specific, cost-sensitive | **AWS FIS** | Native AWS integration, pay-per-experiment |

**For most teams:** Chaos Toolkit (flexible, free) or Gremlin (fast, managed)

## Prerequisites Checklist

Before FIRST experiment:

**Monitoring (Required):**
- [ ] Real-time dashboards for key metrics (latency, error rate, throughput)
- [ ] Distributed tracing for request flows
- [ ] Log aggregation with timeline correlation
- [ ] Alerts configured with thresholds

**Rollback (Required):**
- [ ] Automated rollback based on metrics (e.g., error rate > 5% → abort)
- [ ] Manual kill switch everyone can activate
- [ ] Rollback tested and documented (< 30 sec recovery)

**Baseline (Required):**
- [ ] Documented normal metrics (P50/P95/P99 latency, error rate %)
- [ ] Known dependencies and critical paths
- [ ] System architecture diagram

**Team (Required):**
- [ ] Designated observer monitoring experiment
- [ ] On-call engineer available
- [ ] Communication channel established (war room, Slack)
- [ ] Post-experiment debrief scheduled

## Anti-Patterns Catalog

### ❌ Production First Chaos
**Symptom:** "Let's start chaos testing in production to see what breaks"

**Why bad:** No practice, no muscle memory, production incidents guaranteed

**Fix:** Run 5-10 experiments in staging FIRST. Graduate to production only after proving: experiments work as designed, rollback functions, team can execute response

---

### ❌ Chaos Without Monitoring
**Symptom:** "We injected latency but we're not sure what happened"

**Why bad:** Blind chaos = no learning. You can't validate resilience without seeing system behavior

**Fix:** Set up comprehensive monitoring BEFORE first experiment. Must be able to answer "What changed?" within 30 seconds

---

### ❌ Unlimited Blast Radius
**Symptom:** Affecting 100% of traffic/all services on first run

**Why bad:** Cascading failures, actual outages, customer impact

**Fix:** Start at 0.1-1% traffic. Progression: 0.1% → 1% → 5% → 10% → (stop or 50%). Each step validates before expanding

---

### ❌ Chaos Without Rollback
**Symptom:** "The experiment broke everything and we can't stop it"

**Why bad:** Chaos becomes real incident, 2+ hour recovery, lost trust

**Fix:** Automated abort criteria (error rate threshold, latency threshold, manual kill switch). Test rollback before injecting failures

---

### ❌ Random Chaos (No Hypothesis)
**Symptom:** "Let's inject some failures and see what happens"

**Why bad:** No learning objective, can't validate resilience, wasted time

**Fix:** Every experiment needs hypothesis: "System will [expected behavior] when [failure injected]"

## Failure Types Catalog

Priority order for microservices:

| Failure Type | Priority | Why Test This | Example |
|--------------|----------|---------------|---------|
| **Network Latency** | HIGH | Most common production issue | 500ms delay service A → B |
| **Service Timeout** | HIGH | Tests circuit breakers, retry logic | Service B unresponsive |
| **Connection Loss** | HIGH | Tests failover, graceful degradation | TCP connection drops |
| **Resource Exhaustion** | MEDIUM | Tests resource limits, scaling | Memory limit, connection pool full |
| **Packet Loss** | MEDIUM | Tests retry strategies | 1-10% packet loss |
| **DNS Failure** | MEDIUM | Tests service discovery resilience | DNS resolution delays |
| **Cache Failure** | MEDIUM | Tests fallback behavior | Redis down |
| **Database Errors** | LOW (start) | High risk - test after basics work | Connection refused, query timeout |

**Start with network latency** - safest, most informative, easiest rollback.

## Experiment Template

Use this for every chaos experiment:

**1. Hypothesis**
"If [failure injected], system will [expected behavior], and [metric] will remain [threshold]"

Example: "If service-payment experiences 2s latency, circuit breaker will open within 10s, and P99 latency will stay < 500ms"

**2. Baseline Metrics**
- Current P50/P95/P99 latency:
- Current error rate:
- Current throughput:

**3. Experiment Config**
- Failure type: [latency / packet loss / service down / etc.]
- Target: [specific service / % of traffic]
- Blast radius: [0.1% traffic, single region, canary pods]
- Duration: [2-5 minutes initial]
- Abort criteria: [error rate > 5% OR P99 > 1s OR manual stop]

**4. Execution**
- Observer: [name] monitoring dashboards
- Runner: [name] executing experiment
- Kill switch: [procedure]
- Start time: [timestamp]

**5. Observation**
- What happened vs hypothesis:
- Actual metrics during chaos:
- System behavior notes:

**6. Validation**
- ✓ Hypothesis validated / ✗ Hypothesis failed
- Unexpected findings:
- Action items:

## Blast Radius Progression

Safe scaling path:

| Step | Traffic Affected | Duration | Abort If |
|------|------------------|----------|----------|
| **1. Staging** | 100% staging | 5 min | Any production impact |
| **2. Canary** | 0.1% production | 2 min | Error rate > 1% |
| **3. Small** | 1% production | 5 min | Error rate > 2% |
| **4. Medium** | 5% production | 10 min | Error rate > 5% |
| **5. Large** | 10% production | 15 min | Error rate > 5% |

**Never skip steps.** Each step validates before expanding.

**Stop at 10-20% for most experiments** - no need to chaos 100% of production traffic.

**Low-traffic services (< 1000 req/day):** Use absolute request counts instead of percentages. Minimum 5-10 affected requests per step. Example: 100 req/day service should still start with 5-10 requests (6 hours), not 0.1% (1 request every 10 days).

## Your First Experiment (Staging)

**Goal:** Build confidence, validate monitoring, test rollback

**Experiment:** Network latency on non-critical service

```bash
# Example with Chaos Toolkit
1. Pick least critical service (e.g., recommendation engine, not payment)
2. Inject 500ms latency to 100% of staging traffic
3. Duration: 5 minutes
4. Expected: Timeouts handled gracefully, fallback behavior activates
5. Monitor: Error rate, latency, downstream services
6. Abort if: Error rate > 10% or cascading failures
7. Debrief: What did we learn? Did monitoring catch it? Did rollback work?
```

**Success criteria:** You can answer "Did our hypothesis hold?" within 5 minutes of experiment completion.

## Common Mistakes

### ❌ Testing During Incidents
**Fix:** Only chaos test during stable periods, business hours, with extra staffing

---

### ❌ Network Latency Underestimation
**Fix:** Latency cascades - 500ms can become 5s downstream. Start with 100-200ms, observe, then increase

---

### ❌ No Post-Experiment Review
**Fix:** Every experiment gets 15-min debrief: What worked? What broke? What did we learn?

## Quick Reference

**Prerequisites Before First Chaos:**
1. Monitoring + alerts
2. Automated rollback
3. Baseline metrics documented
4. Team coordinated

**Experiment Steps:**
1. Write hypothesis
2. Document baseline
3. Define blast radius (start 0.1%)
4. Set abort criteria
5. Execute with observer
6. Validate hypothesis
7. Debrief team

**Blast Radius Progression:**
Staging → 0.1% → 1% → 5% → 10% (stop for most experiments)

**First Experiment:**
Network latency (500ms) on non-critical service in staging for 5 minutes

## Bottom Line

**Chaos engineering is hypothesis-driven science, not random destruction.**

Start small (staging, 0.1% traffic), with monitoring, with rollback. Graduate slowly.
