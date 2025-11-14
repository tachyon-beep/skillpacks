---
name: testing-in-production
description: Use when validating changes safely in production through feature flags, canary releases, blue-green deployments, shadow traffic, progressive rollout, and instant rollback strategies
---

# Testing in Production

## Overview

Staging never matches production - different data, different traffic patterns, different scale. Testing in production validates changes with real users, real data, real load - while minimizing risk through controlled rollout and instant rollback.

**Core Principle**: Start small (1% traffic), monitor closely, expand gradually, maintain instant rollback. Feature flags for control, canary releases for validation, observability for detection.

**Ordis Identity**: Testing in production is controlled exposure - systematic validation with real conditions while maintaining defensive layers (rollback, monitoring, blast radius control).

## When to Use

**Use this skill when**:
- Deploying risky changes to production
- Staging doesn't catch production-specific issues
- Need to validate with real traffic patterns
- Implementing progressive rollout strategies
- A/B testing new features
- Blue-green or canary deployments

**Don't use for**:
- Untested code (test in pre-prod first)
- Breaking changes without rollback plan
- Systems without monitoring

## Feature Flags

**Feature flags**: Runtime configuration to enable/disable features without deployment.

### Basic Feature Flag

```javascript
// Feature flag check
if (featureFlags.isEnabled('new-checkout')) {
  return newCheckout Process(cart);
} else {
  return oldCheckoutProcess(cart);
}
```

### User-Targeted Flags

**Gradual rollout by percentage**:

```javascript
// Roll out to 10% of users
if (featureFlags.isEnabledForUser('new-checkout', userId, { rollout: 0.10 })) {
  return newCheckoutProcess(cart);
} else {
  return oldCheckoutProcess(cart);
}
```

**Implementation**:
```javascript
function isEnabledForUser(flagName, userId, options = {}) {
  // Hash user ID to get consistent assignment
  const hash = murmurhash(userId + flagName);
  const bucket = hash % 100;  // 0-99

  return bucket < (options.rollout * 100);
}

// User "user_123" always gets same result for "new-checkout"
// If rollout=0.10, users in buckets 0-9 see new feature
```

### Targeting Strategies

```javascript
// Internal employees only
if (featureFlags.isEnabled('experimental-feature', { targetEmployees: true })) {
  // ...
}

// Specific user IDs (beta testers)
if (featureFlags.isEnabled('beta-feature', { userIds: ['user_1', 'user_2'] })) {
  // ...
}

// Geographic targeting
if (featureFlags.isEnabled('region-feature', { region: 'us-west' })) {
  // ...
}
```

**Tools**: LaunchDarkly, Unleash, Split.io, Flagsmith, or custom

## Canary Releases

**Canary**: Deploy new version to small percentage, monitor, expand gradually.

### Deployment Flow

```
Deploy v2.0.0:

Step 1: 1% traffic → v2.0.0, 99% → v1.9.0 (5 minutes)
  Monitor: error rate, latency, business metrics
  Decision: Proceed or rollback?

Step 2: 10% traffic → v2.0.0, 90% → v1.9.0 (15 minutes)
  Monitor: error rate, latency, business metrics
  Decision: Proceed or rollback?

Step 3: 50% traffic → v2.0.0, 50% → v1.9.0 (30 minutes)
  Monitor: error rate, latency, business metrics
  Decision: Proceed or rollback?

Step 4: 100% traffic → v2.0.0
  Complete deployment
```

### Kubernetes Canary Example

```yaml
# v1 deployment (stable)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-stable
spec:
  replicas: 9  # 90% traffic
  selector:
    matchLabels:
      app: api
      version: v1
---
# v2 deployment (canary)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-canary
spec:
  replicas: 1  # 10% traffic
  selector:
    matchLabels:
      app: api
      version: v2
---
# Service routes to both versions
apiVersion: v1
kind: Service
metadata:
  name: api
spec:
  selector:
    app: api  # Matches both v1 and v2
```

**Traffic split**: 1 canary pod / 10 total pods = 10% canary traffic

### Automated Canary Analysis

```javascript
// Automated canary validation
async function validateCanary(canaryVersion) {
  const metrics = await getMetrics({
    timeRange: '5m',
    versions: ['stable', 'canary']
  });

  // Compare canary to stable
  const canaryErrorRate = metrics.canary.errorRate;
  const stableErrorRate = metrics.stable.errorRate;

  if (canaryErrorRate > stableErrorRate * 1.5) {
    // Canary error rate 50% higher than stable
    await rollbackCanary();
    throw new Error('Canary failed: high error rate');
  }

  const canaryP95 = metrics.canary.latencyP95;
  const stableP95 = metrics.stable.latencyP95;

  if (canaryP95 > stableP95 * 1.2) {
    // Canary latency 20% higher than stable
    await rollbackCanary();
    throw new Error('Canary failed: high latency');
  }

  // All checks passed
  return true;
}
```

**Tools**: Flagger (Kubernetes), Spinnaker, Argo Rollouts

## Blue-Green Deployments

**Blue-Green**: Two identical environments, switch traffic instantly.

```
Current state:
  Blue (v1.0) ← 100% production traffic
  Green (idle)

Deploy v2.0:
1. Deploy v2.0 to Green environment
2. Test Green environment (smoke tests)
3. Switch traffic: Blue → Green
4. Monitor for issues
5. If problems: Switch traffic back to Blue
6. If success: Blue becomes idle (ready for next deploy)
```

### Load Balancer Switch

```
# Before deployment
Load Balancer → Blue (v1.0)

# After deployment
Load Balancer → Green (v2.0)

# If rollback needed
Load Balancer → Blue (v1.0) (instant rollback)
```

### Implementation (AWS/Cloud)

```bash
# Deploy to green environment
terraform apply -var="active_env=green"

# Switch DNS/load balancer
aws elbv2 modify-rule \
  --rule-arn $RULE_ARN \
  --actions Type=forward,TargetGroupArn=$GREEN_TARGET_GROUP

# Monitor for 15 minutes

# If success, green is now production
# If failure, switch back to blue
aws elbv2 modify-rule \
  --rule-arn $RULE_ARN \
  --actions Type=forward,TargetGroupArn=$BLUE_TARGET_GROUP
```

**Advantages**:
- Instant rollback (switch traffic back)
- Zero downtime
- Full environment testing before cutover

**Disadvantages**:
- 2x infrastructure cost (two full environments)
- Database migrations tricky (must work with both versions)

## Shadow Traffic

**Shadow traffic**: Send copy of production traffic to new version, compare results.

```
Production traffic:
  → v1.0 (serves real users)
  → v2.0 (receives copy, results discarded)

Compare:
- Response times
- Error rates
- Output differences
```

### Implementation

```javascript
// Proxy that shadows traffic
app.post('/api/checkout', async (req, res) => {
  // Primary: Send to production (v1)
  const primaryResult = await v1Service.checkout(req.body);

  // Shadow: Send to canary (v2) - fire and forget
  shadowRequest(v2Service.checkout, req.body).catch(err => {
    // Log differences, don't affect user
    logger.warn('Shadow request difference', { err });
  });

  // Return primary result to user
  res.json(primaryResult);
});

async function shadowRequest(fn, data) {
  const shadowResult = await fn(data);

  // Compare results (for analysis)
  if (JSON.stringify(shadowResult) !== JSON.stringify(primaryResult)) {
    metrics.increment('shadow.difference');
    logger.warn('Shadow result differs from primary');
  }
}
```

**Use case**: Validate new implementation before exposing to users.

## Progressive Rollout

**Progressive rollout**: Gradually increase traffic to new version.

### Rollout Schedule

```
Day 1: 1% of users
  Monitor: 24 hours
  Abort if: error rate > 1%

Day 2: 10% of users
  Monitor: 24 hours
  Abort if: error rate > 0.5%

Day 3: 25% of users
  Monitor: 24 hours
  Abort if: error rate > 0.1%

Day 4: 50% of users
  Monitor: 24 hours

Day 5: 100% of users
  Complete rollout
```

### Feature Flag Automation

```javascript
// Automated progressive rollout
async function progressiveRollout(flagName) {
  const schedule = [
    { percentage: 1, duration: '24h' },
    { percentage: 10, duration: '24h' },
    { percentage: 25, duration: '24h' },
    { percentage: 50, duration: '24h' },
    { percentage: 100, duration: '∞' }
  ];

  for (const step of schedule) {
    await featureFlags.setRollout(flagName, step.percentage / 100);

    console.log(`Rolled out to ${step.percentage}%`);

    // Monitor metrics
    await sleep(step.duration);

    const metrics = await getMetrics({ flag: flagName });

    if (metrics.errorRate > 0.01) {  // >1% error rate
      await featureFlags.setRollout(flagName, 0);  // Rollback
      throw new Error('Rollout failed: high error rate');
    }
  }

  console.log('Progressive rollout complete');
}
```

## Rollback Strategies

### Instant Rollback (Feature Flag)

```javascript
// Disable feature flag immediately
await featureFlags.disable('new-checkout');

// All users revert to old code instantly (no deployment)
```

**Fastest rollback**: Seconds (no deployment needed)

### Deployment Rollback

```bash
# Kubernetes rollback
kubectl rollout undo deployment/api

# Rollback to specific version
kubectl rollout undo deployment/api --to-revision=5
```

**Speed**: Minutes (redeploy previous version)

### Blue-Green Rollback

```bash
# Switch load balancer back to blue
aws elbv2 modify-rule --rule-arn $RULE_ARN \
  --actions Type=forward,TargetGroupArn=$BLUE_TARGET_GROUP
```

**Speed**: Seconds (DNS/LB switch)

## A/B Testing for Quality

**A/B testing**: Compare two versions with real users.

```javascript
// A/B test: new algorithm vs old
const variant = abTest.getVariant(userId, 'recommendation-algo');

if (variant === 'A') {
  recommendations = oldAlgorithm(userId);
} else {
  recommendations = newAlgorithm(userId);
}

// Track metrics by variant
metrics.track('recommendations.shown', { variant });
```

**Analyze**:
```sql
SELECT
  variant,
  AVG(click_rate) as avg_click_rate,
  AVG(conversion_rate) as avg_conversion_rate
FROM metrics
WHERE experiment = 'recommendation-algo'
GROUP BY variant

-- Results:
-- A (old): 3.5% click rate, 1.2% conversion
-- B (new): 4.2% click rate, 1.5% conversion  ← Winner
```

## Monitoring Requirements

**Before testing in production**, must have:

✅ Real-time metrics dashboards
✅ Error rate alerts
✅ Latency alerts (p95, p99)
✅ Business metric tracking (orders/sec, signups)
✅ Comparison metrics (canary vs stable)
✅ Automated rollback triggers

## Quick Reference

| Strategy | Rollout Speed | Rollback Speed | Risk | Cost |
|----------|---------------|----------------|------|------|
| **Feature flag** | Instant | Instant | Low | Low |
| **Canary** | Gradual (hours) | Fast (minutes) | Low | Medium |
| **Blue-green** | Instant switch | Instant | Medium | High (2x infra) |
| **Progressive** | Gradual (days) | Instant (flag) | Very low | Low |
| **Shadow** | No user impact | N/A | Zero | Low |

| Rollout % | Monitor Duration | Abort Threshold |
|-----------|------------------|-----------------|
| **1%** | 5-15 minutes | Error rate > 5% |
| **10%** | 30-60 minutes | Error rate > 1% |
| **25%** | 2-4 hours | Error rate > 0.5% |
| **50%** | 8-12 hours | Error rate > 0.1% |
| **100%** | Ongoing | SLO violation |

## Common Mistakes

### ❌ No Rollback Plan

**Wrong**: Deploy to production, hope it works
**Right**: Feature flag or blue-green for instant rollback

**Why**: Things go wrong - need instant abort.

### ❌ Big Bang Rollout

**Wrong**: Deploy to 100% of users immediately
**Right**: 1% → 10% → 50% → 100%

**Why**: Catch issues with small blast radius.

### ❌ No Monitoring

**Wrong**: Deploy canary, check manually later
**Right**: Real-time dashboards, automated alerts

**Why**: Can't detect issues without monitoring.

### ❌ Testing Untested Code in Production

**Wrong**: Skip staging, test only in production
**Right**: Test in staging first, then production validation

**Why**: Production testing is for validation, not initial testing.

### ❌ Ignoring Metrics

**Wrong**: Canary shows high error rate, proceed anyway
**Right**: Automated rollback if metrics degrade

**Why**: Metrics exist to prevent bad deployments.

## Real-World Impact

**Before Testing in Production**:
- Big bang deployments (100% rollout)
- Issues discovered after full deployment
- Long rollback time (redeploy previous version)
- Multiple production incidents

**After Testing in Production**:
- Progressive rollout (1% → 100% over days)
- Feature flags for instant rollback
- Issues caught at 1% rollout (minimal impact)
- Automated canary analysis
- Zero production incidents from bad deployments

## Summary

**Testing in production validates changes safely with real conditions:**

1. **Feature flags** (instant enable/disable without deployment)
2. **Canary releases** (1% → 10% → 50% → 100%)
3. **Blue-green** (instant switch between environments)
4. **Progressive rollout** (gradual expansion over days)
5. **Shadow traffic** (validate without user impact)
6. **Monitor closely** (real-time metrics, automated rollback)
7. **Instant rollback** (flags, DNS switch, deployment rollback)

**Ordis Principle**: Production testing is controlled exposure - systematic validation with defensive layers (rollback, monitoring, gradual expansion).
