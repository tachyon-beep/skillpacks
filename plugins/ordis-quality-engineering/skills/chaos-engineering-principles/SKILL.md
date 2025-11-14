---
name: chaos-engineering-principles
description: Use when testing system resilience through controlled failure injection - provides steady-state hypothesis definition, blast radius limiting, production chaos patterns, and learning from failures
---

# Chaos Engineering Principles

## Overview

Chaos engineering is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions in production. You inject failures deliberately to find weaknesses before they cause outages.

**Core Principle**: Define steady-state first, inject failures in controlled experiments, start small (blast radius), expand gradually. Production chaos requires observability and kill switches.

**Ordis Identity**: Chaos engineering tests the bulwark by attacking it systematically - controlled disorder to strengthen defenses before real chaos strikes.

## When to Use

**Use this skill when**:
- Testing system resilience and fault tolerance
- Validating incident response procedures
- Finding single points of failure
- Planning Game Days or fire drills
- Verifying circuit breakers, retries, failover work
- Building confidence in distributed systems
- Questions like "what if database fails?", "what if network is slow?"

**Don't use for**:
- Systems without monitoring (you won't see impact)
- Untested code (fix bugs first, then test resilience)
- Single-instance systems with no redundancy (nothing to test)

## Principles of Chaos Engineering

### 1. Define Steady-State Hypothesis

**Before** injecting failures, define what "working" looks like.

**Wrong**:
```
"Let's kill the database and see what happens"
```

**Right**:
```
Steady-state hypothesis:
- 99% of requests return 200 status
- p99 latency < 500ms
- Order creation succeeds
- Dashboard shows order count increasing

Expected when database fails:
- System uses cache/fallback
- Orders queue for later processing
- Users see degraded experience but no errors
```

**Metrics to monitor**:
- Request success rate
- Latency (p95, p99)
- Error rate
- Business metrics (orders/sec, signups/hour)

### 2. Inject Realistic Failures

**Common failure types**:

**Network failures**:
- Latency injection (slow network)
- Packet loss (unreliable network)
- Network partition (split-brain)
- DNS failures

**Compute failures**:
- Process crashes (kill application)
- CPU exhaustion (stress CPU to 100%)
- Memory exhaustion (consume all memory)
- Disk full

**State failures**:
- Database unavailable
- Cache miss/eviction
- File system errors
- Clock skew (time drift)

### 3. Run Experiments in Production

**Why production?**
- Staging doesn't have real traffic patterns
- Production has real data, real integrations
- Only production shows real user impact

**Safety requirements**:
- Monitoring and alerting in place
- Kill switch to abort experiment
- Blast radius limiting (affect subset of traffic)
- On-call engineer ready
- Stakeholders notified

### 4. Automate Experiments

**Manual chaos**: One-off experiments
**Automated chaos**: Continuous verification

**Example**: GameDay events → Automated weekly chaos tests

### 5. Minimize Blast Radius

**Start small, expand gradually:**

```
Experiment 1: Single instance, 1% traffic, 5 minutes
↓ (if successful)
Experiment 2: Multiple instances, 10% traffic, 30 minutes
↓ (if successful)
Experiment 3: Full production, 50% traffic, 2 hours
```

## Chaos Experiments

### Experiment Template

```yaml
name: database-failure-experiment
hypothesis: |
  When primary database fails, system uses read replica
  with <5% error rate and p99 latency <1000ms

steady-state:
  metrics:
    - error_rate < 0.01
    - p99_latency < 500ms
    - orders_per_minute > 100

blast_radius:
  region: us-west-2
  percentage: 10%  # Affect 10% of traffic
  duration: 15m

failure_injection:
  type: network_partition
  target: primary_database
  duration: 10m

abort_conditions:
  - error_rate > 0.10  # >10% errors → abort
  - p99_latency > 2000ms

rollback:
  - restore network
  - restart database connections
```

### Example: Database Failure

```python
# Chaos experiment: Database failure
import time
from chaos_toolkit.experiment import run_experiment

experiment = {
    "title": "Database resilience test",
    "description": "Verify application handles database failure gracefully",

    "steady-state-hypothesis": {
        "title": "System is healthy",
        "probes": [
            {
                "type": "probe",
                "name": "API is responding",
                "tolerance": {
                    "type": "probe",
                    "name": "less_than_5_percent_errors",
                    "provider": {
                        "type": "python",
                        "func": "check_error_rate",
                        "arguments": {"threshold": 0.05}
                    }
                }
            }
        ]
    },

    "method": [
        {
            "type": "action",
            "name": "Terminate database connections",
            "provider": {
                "type": "python",
                "module": "chaosaws.rds.actions",
                "func": "stop_instance",
                "arguments": {
                    "instance_id": "prod-db-primary",
                    "region": "us-west-2"
                }
            },
            "pauses": {"after": 60}  # Wait 1 min
        },
        {
            "type": "probe",
            "name": "Verify failover to replica",
            "provider": {
                "type": "python",
                "func": "check_database_connection",
                "arguments": {"expected": "replica"}
            }
        }
    ],

    "rollbacks": [
        {
            "type": "action",
            "name": "Restart database",
            "provider": {
                "type": "python",
                "module": "chaosaws.rds.actions",
                "func": "start_instance",
                "arguments": {"instance_id": "prod-db-primary"}
            }
        }
    ]
}

# Run with abort conditions monitoring
run_experiment(experiment)
```

## Chaos Tools

### Chaos Mesh (Kubernetes)

**Best for**: Kubernetes-based systems

```yaml
# ChaosManifest for pod failure
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-failure-test
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - production
    labelSelectors:
      app: api-server
  duration: "30s"
  scheduler:
    cron: "@every 1h"  # Run hourly
```

**Failure types**: Pod kill, network delay, IO faults, CPU/memory stress

### Litmus Chaos (Cloud-Native)

**Best for**: Cloud-native applications

```yaml
# Litmus experiment: Node CPU stress
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: cpu-stress
spec:
  engineState: 'active'
  appinfo:
    appns: 'production'
    applabel: 'app=api'
  chaosServiceAccount: litmus-admin
  experiments:
    - name: node-cpu-hog
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: '60'
            - name: CPU_CORES
              value: '2'
```

### Gremlin (Commercial)

**Best for**: Enterprise with support needs, non-Kubernetes

**Advantages**: GUI, scheduled experiments, enterprise features
**Disadvantages**: Commercial license required

### Chaos Toolkit (Open Source)

**Best for**: Custom experiments, multi-cloud

```python
# Python-based experiments (example above)
```

**Recommendation**: Use Chaos Mesh for Kubernetes, Chaos Toolkit for flexibility.

## Production Chaos Safety

### Prerequisites

**Before running chaos in production**:

✅ Monitoring and alerting configured
✅ SLOs defined
✅ On-call engineer available
✅ Stakeholders notified
✅ Rollback plan tested
✅ Blast radius limited (percentage of traffic)
✅ Kill switch ready

### Feature Flags for Blast Radius

```python
# Use feature flags to limit blast radius
if feature_flag.is_enabled('chaos-experiment', user_id):
    # This user gets chaos-injected traffic
    database = degraded_database  # Slow queries
else:
    # Normal traffic
    database = primary_database
```

**Benefits**:
- Control exactly which users affected
- Gradual rollout (1% → 10% → 50%)
- Instant abort (disable flag)

### Kill Switch

```python
# Monitoring-based kill switch
@every_minute
def check_experiment_health():
    error_rate = metrics.get_error_rate()
    latency_p99 = metrics.get_p99_latency()

    if error_rate > 0.10:  # >10% errors
        abort_experiment()
        alert("Chaos experiment aborted: high error rate")

    if latency_p99 > 2000:  # p99 > 2s
        abort_experiment()
        alert("Chaos experiment aborted: high latency")
```

## Game Days

**Game Day**: Scheduled chaos engineering event with team participation.

### Planning

```
1. Schedule (2-3 weeks notice)
   - Date/time
   - Participants (eng, SRE, product)
   - On-call engineer

2. Define scenarios
   - Database failure
   - Region outage
   - Third-party API down
   - Traffic spike

3. Prepare
   - Test scenarios in staging
   - Set up monitoring dashboards
   - Create runbooks
   - Notify stakeholders

4. Execute
   - Run scenarios in production
   - Monitor metrics
   - Document findings
   - Practice incident response

5. Debrief
   - What broke?
   - What worked?
   - Action items
   - Update runbooks
```

### Example Game Day Agenda

```
9:00 AM - Kickoff
  - Review scenarios
  - Review abort conditions

9:30 AM - Scenario 1: Database Failure
  - Inject failure
  - Monitor metrics
  - Verify failover
  - Document findings

10:30 AM - Break

10:45 AM - Scenario 2: Network Latency
  - Inject 500ms latency
  - Observe circuit breakers
  - Check timeout behavior

11:45 AM - Scenario 3: Traffic Spike
  - 10x traffic for 15 minutes
  - Monitor autoscaling
  - Check rate limiting

12:45 PM - Debrief
  - Review findings
  - Action items
  - Update runbooks
```

## Learning from Chaos

### Document Findings

```markdown
# Chaos Experiment Report

## Experiment: Database Failure
**Date**: 2025-01-15
**Duration**: 15 minutes
**Blast Radius**: 10% of production traffic

## Hypothesis
System uses read replica when primary database fails,
with <5% error rate and p99 latency <1000ms.

## Results
- ❌ Error rate spiked to 25% (hypothesis violated)
- ❌ p99 latency: 5000ms (hypothesis violated)
- ✅ System eventually recovered (circuit breaker triggered)

## Root Cause
- Connection pool not configured for replica
- 30-second timeout before circuit breaker triggered
- Retry logic caused cascading failures

## Action Items
1. Configure connection pool for replica failover
2. Reduce circuit breaker timeout to 5 seconds
3. Implement exponential backoff for retries
4. Add monitoring for connection pool usage

## Impact
- 10% of users saw errors for 2 minutes
- No data loss
- Automatic recovery after 3 minutes
```

### Iterate

```
Experiment 1: Database failure → Found connection pool issue
  ↓
Fix: Configure replica connection pool
  ↓
Experiment 2: Database failure → Found slow circuit breaker
  ↓
Fix: Reduce timeout to 5 seconds
  ↓
Experiment 3: Database failure → Success (hypothesis validated)
```

## Common Chaos Scenarios

| Scenario | Failure Type | What to Test |
|----------|--------------|--------------|
| **Database down** | Network partition | Failover to replica, caching, graceful degradation |
| **High latency** | Network delay | Timeouts, circuit breakers, retry logic |
| **Pod crash** | Process kill | Kubernetes restarts, health checks, load balancing |
| **Disk full** | Resource exhaustion | Monitoring alerts, log rotation, cleanup |
| **Region outage** | Multi-AZ failure | Cross-region failover, DNS routing |
| **Third-party API down** | Network block | Fallback behavior, cached responses, error handling |
| **Memory leak** | Memory stress | OOM handling, restart policies, alerting |
| **CPU exhaustion** | CPU stress | Autoscaling, load shedding, rate limiting |

## Quick Reference

| Principle | What It Means |
|-----------|---------------|
| **Steady-state first** | Define "working" before breaking things |
| **Start small** | 1% traffic, 5 min duration, single instance |
| **Expand gradually** | 1% → 10% → 50% if experiments succeed |
| **Automate** | Manual Game Days → Automated continuous chaos |
| **Production chaos** | Staging doesn't reveal real issues |
| **Kill switch** | Abort if error rate or latency exceeds threshold |
| **Learn and iterate** | Document findings, fix issues, re-test |

## Common Mistakes

### ❌ No Steady-State Hypothesis

**Wrong**: "Let's break things and see what happens"
**Right**: "Define success criteria, inject failure, verify hypothesis"

**Why**: Without hypothesis, you don't know if experiment succeeded.

### ❌ Testing in Staging Only

**Wrong**: "We tested failover in staging, we're good"
**Right**: "Staging validated approach, now test in production with 1% traffic"

**Why**: Production has real traffic, real integrations, real data - staging doesn't.

### ❌ No Blast Radius Control

**Wrong**: "Let's kill the database in all regions"
**Right**: "Let's kill database for 1% of traffic in one region"

**Why**: Uncontrolled chaos = production outage, not experiment.

### ❌ No Rollback Plan

**Wrong**: "If it breaks, we'll figure it out"
**Right**: "Rollback script ready, tested in staging, kill switch configured"

**Why**: Experiments sometimes go wrong - need instant abort.

### ❌ Not Monitoring

**Wrong**: Run chaos, check logs manually
**Right**: Real-time dashboards, automated alerts, metrics comparison

**Why**: Can't assess impact without observability.

### ❌ Ignoring Findings

**Wrong**: "Database failover didn't work, noted"
**Right**: "Database failover failed → action item → fix → re-test"

**Why**: Chaos reveals weaknesses - must fix them.

## Real-World Impact

**Before Chaos Engineering**:
- Assumed database failover worked (never tested)
- Production database outage
- 2-hour recovery time
- Manual failover process
- Lost revenue

**After Chaos Engineering**:
- Weekly automated chaos tests
- Found connection pool misconfiguration
- Fixed timeout issues
- Validated circuit breakers
- Next database outage: automatic failover in 30 seconds

## Summary

**Chaos engineering tests resilience through controlled failure injection:**

1. **Define steady-state hypothesis** (what "working" looks like)
2. **Start small** (1% traffic, 5 min, single instance)
3. **Expand gradually** if successful
4. **Production chaos** (staging doesn't reveal real issues)
5. **Safety first** (monitoring, kill switch, rollback plan)
6. **Learn and iterate** (fix issues, re-test, automate)
7. **Game Days** (team practice for incident response)

**Ordis Principle**: Test the bulwark by attacking it - controlled chaos strengthens defenses before real siege.
