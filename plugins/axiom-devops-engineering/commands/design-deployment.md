---
description: Design deployment strategy with zero-downtime, rollback capability, and verification gates
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[application_or_service_name]"
---

# Design Deployment Command

You are designing a deployment strategy that ensures zero-downtime, automatic rollback, and production safety.

## Core Principle

**"Deploy to production" is not a single step - it's a sequence of gates, health checks, gradual rollouts, and automated rollback triggers.**

## Information Gathering

Before designing, determine:

1. **Infrastructure**: Kubernetes, containers, VMs, serverless?
2. **Traffic volume**: Requests per second, peak load?
3. **Risk tolerance**: Can you afford double infrastructure temporarily?
4. **Database**: Stateful migrations required?
5. **Current state**: Existing CI/CD, deployment process?

## Deployment Strategy Selection

### Decision Matrix

| Factor | Blue-Green | Canary | Rolling |
|--------|------------|--------|---------|
| **Best for** | Critical systems | High-traffic | Cost-sensitive |
| **Infrastructure cost** | 2x during deploy | 1.1x during deploy | 1x |
| **Rollback speed** | Instant | Fast | Gradual |
| **Complexity** | Medium | High | Low |
| **Traffic control** | All-or-nothing | Percentage-based | Instance-based |
| **Risk exposure** | Low | Very low | Medium |

### Blue-Green Deployment

**How it works**:
```
Old (Blue) ← 100% traffic
New (Green) ← deployed, verified, 0% traffic

→ Health check Green
→ Switch traffic to Green (instant)
→ Keep Blue running 1 hour
→ If issues: switch back to Blue (instant rollback)
→ Terminate Blue when stable
```

**Use when**:
- Instant rollback is critical
- Can afford temporary double infrastructure
- Simple traffic switching available (load balancer, K8s service)

**Implementation**:
```yaml
# Kubernetes example
deploy_green:
  - Deploy new pods with label: version=green
  - Wait for health checks to pass
  - Run smoke tests against green (internal)

cutover:
  - Update service selector to version=green
  - Monitor for 15 minutes

rollback:
  - Update service selector to version=blue
  - Investigate issues

cleanup:
  - After 1 hour stable, delete blue pods
```

### Canary Deployment

**How it works**:
```
Old ← 95% traffic
New (Canary) ← 5% traffic

→ Monitor error rates, latency for 15 min
→ If healthy: shift to 25%
→ If healthy: shift to 50%
→ If healthy: shift to 100%
→ If unhealthy at any stage: immediate 100% to old
```

**Use when**:
- High traffic volume (enough for statistical significance)
- Good observability (metrics, logging)
- Want gradual risk exposure

**Implementation**:
```yaml
canary_stages:
  - percentage: 5
    duration: 15m
    success_criteria:
      error_rate: < 1%
      p99_latency: < 500ms

  - percentage: 25
    duration: 15m
    success_criteria:
      error_rate: < 1%
      p99_latency: < 500ms

  - percentage: 50
    duration: 15m

  - percentage: 100

rollback_triggers:
  - error_rate: > 5%
  - p99_latency: > 1000ms
  - health_check_failures: > 2
```

### Rolling Deployment

**How it works**:
```
Instances: [A, B, C, D, E]

→ Deploy to A, health check
→ Deploy to B, health check
→ Deploy to C, D, E sequentially
→ If any fails: stop, rollback deployed instances
```

**Use when**:
- Cost-sensitive (no extra infrastructure)
- Can tolerate brief mixed-version state
- Lower traffic volume

**Implementation**:
```yaml
rolling:
  max_unavailable: 1  # One instance at a time
  max_surge: 0        # No extra instances

  per_instance:
    - Drain connections (30s)
    - Deploy new version
    - Health check (retry 3x)
    - Resume traffic

  on_failure:
    - Stop rollout
    - Rollback completed instances
```

## Complete Deployment Design

### Pipeline Structure

```yaml
stages:
  - build
  - test
  - deploy_staging
  - verify_staging
  - deploy_production
  - verify_production
  - monitor

build:
  steps:
    - Compile/build
    - Create container image
    - Tag with $COMMIT_SHA
    - Push to registry

test:
  parallel:
    - unit_tests
    - integration_tests
    - security_scan

deploy_staging:
  environment: staging
  strategy: [chosen_strategy]

verify_staging:
  automated:
    - health_check
    - smoke_tests
    - integration_tests
  gate: block_on_failure

deploy_production:
  environment: production
  strategy: [chosen_strategy]
  requires:
    - verify_staging: passed
    - manual_approval: optional

verify_production:
  automated:
    - health_check
    - smoke_tests
    - error_rate_check
    - latency_check
  auto_rollback:
    - on: health_check_failure
    - on: error_rate > 5%
    - on: latency > 2x_baseline

monitor:
  duration: 1_hour
  dashboard: deployment_metrics
  alerts: on_anomaly
```

### Health Check Design

```yaml
health_endpoint: /health

checks:
  - name: application_ready
    path: /health/ready
    success: status == 200

  - name: database_connected
    path: /health/db
    success: status == 200

  - name: dependencies_healthy
    path: /health/dependencies
    success: status == 200

probe_config:
  initial_delay: 10s
  interval: 5s
  timeout: 3s
  failure_threshold: 3
  success_threshold: 1
```

### Auto-Rollback Triggers

```yaml
rollback_triggers:
  immediate:
    - health_check_failures >= 3
    - error_rate > 10%

  gradual:
    - error_rate > 5% for 3 minutes
    - p99_latency > 2x baseline for 5 minutes
    - pod_restarts > 3 in 5 minutes

rollback_action:
  blue_green: switch_to_previous_version
  canary: route_100%_to_stable
  rolling: redeploy_previous_version
```

### Database Migration Strategy

```yaml
migration_approach: three_phase

phase_1:
  description: Add new schema (backward compatible)
  changes:
    - Add nullable columns
    - Create new tables
    - Add indexes with CONCURRENTLY
  rollback: drop_new_elements

phase_2:
  description: Deploy code using both schemas
  code_changes:
    - Write to both old and new
    - Read from new with fallback
  rollback: revert_code

phase_3:
  description: Remove old schema
  changes:
    - Drop old columns
    - Remove old tables
  timing: after_phase_2_stable_1_week
```

## Output Format

```markdown
# Deployment Design: [Application Name]

## Context

**Infrastructure**: [K8s/Containers/VMs/Serverless]
**Traffic**: [Requests/second]
**Risk Tolerance**: [High/Medium/Low]
**Current State**: [Existing process]

## Chosen Strategy

**Strategy**: [Blue-Green/Canary/Rolling]
**Rationale**: [Why this strategy fits]

## Pipeline Design

### Stages

```yaml
[Complete pipeline YAML]
```

### Deployment Configuration

```yaml
[Strategy-specific configuration]
```

### Health Checks

```yaml
[Health check configuration]
```

### Auto-Rollback

```yaml
[Rollback trigger configuration]
```

## Database Migrations

**Approach**: [Three-phase/Zero-downtime/N/A]

[Migration steps if applicable]

## Verification Gates

### Staging Verification
- [ ] Health endpoint returns 200
- [ ] Smoke tests pass
- [ ] No error rate increase

### Production Verification
- [ ] Health checks pass
- [ ] Error rate < 1%
- [ ] Latency within baseline
- [ ] Business metrics normal

## Rollback Procedure

**Trigger**: [Conditions]
**Action**: [Steps]
**Verification**: [How to confirm rollback worked]

## Implementation Checklist

- [ ] Health check endpoint implemented
- [ ] Staging environment matches production
- [ ] Monitoring dashboard configured
- [ ] Rollback tested in staging
- [ ] On-call notified of deployment process
- [ ] Documentation updated
```

## Cross-Pack Discovery

```python
import glob

# For monitoring setup
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("Available: ordis-quality-engineering for observability patterns")

# For API deployment
web_pack = glob.glob("plugins/axiom-web-backend/plugin.json")
if web_pack:
    print("Available: axiom-web-backend for API deployment patterns")
```

## Scope Boundaries

**This command covers:**
- Deployment strategy selection
- Pipeline design
- Health check configuration
- Rollback trigger design
- Database migration approach

**Not covered:**
- Pipeline review (use /review-pipeline)
- Infrastructure provisioning
- Application code changes
