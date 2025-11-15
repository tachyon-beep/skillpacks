---
name: testing-in-production
description: Use when implementing feature flags, canary deployments, shadow traffic, A/B testing, choosing blast radius limits, defining rollback criteria, or monitoring production experiments - provides technique selection, anti-patterns, and kill switch frameworks
---

# Testing in Production

## Overview

**Core principle:** Minimize blast radius, maximize observability, always have a kill switch.

**Rule:** Testing in production is safe when you control exposure and can roll back instantly.

## Technique Selection Decision Tree

| Your Goal | Risk Tolerance | Infrastructure Needed | Use |
|-----------|----------------|----------------------|-----|
| Test feature with specific users | Low | Feature flag service | **Feature Flags** |
| Validate deployment safety | Medium | Load balancer, multiple instances | **Canary Deployment** |
| Compare old vs new performance | Low | Traffic duplication | **Shadow Traffic** |
| Measure business impact | Medium | A/B testing framework, analytics | **A/B Testing** |
| Test without any user impact | Lowest | Service mesh, traffic mirroring | **Dark Launch** |

**First technique:** Feature flags (lowest infrastructure requirement, highest control)

## Anti-Patterns Catalog

### ❌ Nested Feature Flags
**Symptom:** Flags controlling other flags, creating combinatorial complexity

**Why bad:** 2^N combinations to test, impossible to validate all paths, technical debt accumulates

**Fix:** Maximum 1 level of flag nesting, delete flags after rollout

```python
# ❌ Bad
if feature_flags.enabled("new_checkout"):
    if feature_flags.enabled("express_shipping"):
        if feature_flags.enabled("gift_wrap"):
            # 8 possible combinations for 3 flags

# ✅ Good
if feature_flags.enabled("new_checkout_v2"):  # Single flag for full feature
    return new_checkout_with_all_options()
```

---

### ❌ Canary with Sticky Sessions
**Symptom:** Users switch between old and new versions across requests due to session affinity

**Why bad:** Inconsistent experience, state corruption, false negative metrics

**Fix:** Route user to same version for entire session

```nginx
# ✅ Good - Consistent routing
upstream backend {
    hash $cookie_user_id consistent;  # Sticky by user ID
    server backend-v1:8080 weight=95;
    server backend-v2:8080 weight=5;
}
```

---

### ❌ No Statistical Validation
**Symptom:** Making rollout decisions on small sample sizes without confidence intervals

**Why bad:** Random variance mistaken for real effects, premature rollback or expansion

**Fix:** Minimum sample size, statistical significance testing

```python
# ✅ Good - Statistical validation
from scipy import stats

def is_safe_to_rollout(control_errors, treatment_errors, min_sample=1000):
    if len(treatment_errors) < min_sample:
        return False, "Insufficient data"

    # Two-proportion z-test
    _, p_value = stats.proportions_ztest(
        [control_errors.sum(), treatment_errors.sum()],
        [len(control_errors), len(treatment_errors)]
    )

    return p_value > 0.05, f"p-value: {p_value}"
```

---

### ❌ Testing Without Rollback
**Symptom:** Deploying feature flags or canaries without instant kill switch

**Why bad:** When issues detected, can't stop impact immediately

**Fix:** Kill switch tested before first production test

---

### ❌ Insufficient Monitoring
**Symptom:** Monitoring only error rates, missing business/user metrics

**Why bad:** Technical success but business failure (e.g., lower conversion)

**Fix:** Monitor technical + business + user experience metrics

## Blast Radius Control Framework

**Progressive rollout schedule:**

| Phase | Exposure | Duration | Abort If | Continue If |
|-------|----------|----------|----------|-------------|
| **1. Internal** | 10-50 internal users | 1-2 days | Any errors | 0 errors, good UX feedback |
| **2. Canary** | 1% production traffic | 4-24 hours | Error rate > +2%, latency > +10% | Metrics stable |
| **3. Small** | 5% production | 1-2 days | Error rate > +5%, latency > +25% | Metrics stable or improved |
| **4. Medium** | 25% production | 2-3 days | Error rate > +5%, latency > +25% | Metrics stable or improved |
| **5. Majority** | 50% production | 3-7 days | Error rate > +5%, business metrics down | Metrics improved |
| **6. Full** | 100% production | Monitor indefinitely | Business metrics drop | Cleanup old code |

**Minimum dwell time:** Each phase needs minimum observation period to catch delayed issues

**Rollback at any phase:** If metrics degrade, revert to previous phase

## Kill Switch Criteria

**Immediate rollback triggers (automated):**

| Metric | Threshold | Why |
|--------|-----------|-----|
| Error rate increase | > 5% above baseline | User impact |
| p99 latency increase | > 50% above baseline | Performance degradation |
| Critical errors (5xx) | > 0.1% of requests | Service failure |
| Business metric drop | > 10% (conversion, revenue) | Revenue impact |

**Warning triggers (manual investigation):**

| Metric | Threshold | Action |
|--------|-----------|--------|
| Error rate increase | 2-5% above baseline | Halt rollout, investigate |
| p95 latency increase | 25-50% above baseline | Monitor closely |
| User complaints | >3 similar reports | Halt rollout, investigate |

**Statistical validation:**

```python
# Sample size for 95% confidence, 80% power
# Minimum 1000 samples per variant for most A/B tests
# For low-traffic features: wait 24-48 hours regardless
```

## Monitoring Quick Reference

**Required metrics (all tests):**

| Category | Metrics | Alert Threshold |
|----------|---------|-----------------|
| **Errors** | Error rate, exception count, 5xx responses | > +5% vs baseline |
| **Performance** | p50/p95/p99 latency, request duration | p99 > +50% vs baseline |
| **Business** | Conversion rate, transaction completion, revenue | > -10% vs baseline |
| **User Experience** | Client errors, page load, bounce rate | > +20% vs baseline |

**Baseline calculation:**

```python
# Collect baseline from previous 7-14 days
baseline_p99 = np.percentile(historical_latencies, 99)
current_p99 = np.percentile(current_latencies, 99)

if current_p99 > baseline_p99 * 1.5:  # 50% increase
    rollback()
```

## Implementation Patterns

### Feature Flags Pattern

```python
# Using LaunchDarkly, Split.io, or similar
from launchdarkly import LDClient, Context

client = LDClient("sdk-key")

def handle_request(user_id):
    context = Context.builder(user_id).build()

    if client.variation("new-checkout", context, default=False):
        return new_checkout_flow(user_id)
    else:
        return old_checkout_flow(user_id)
```

**Best practices:**
- Default to `False` (old behavior) for safety
- Pass user context for targeting
- Log flag evaluations for debugging
- Delete flags within 30 days of full rollout

### Canary Deployment Pattern

```yaml
# Kubernetes with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: my-service
spec:
  hosts:
  - my-service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: my-service
        subset: v2
  - route:
    - destination:
        host: my-service
        subset: v1
      weight: 95
    - destination:
        host: my-service
        subset: v2
      weight: 5
```

### Shadow Traffic Pattern

```python
# Duplicate requests to new service, ignore responses
import asyncio

async def handle_request(request):
    # Primary: serve user from old service
    response = await old_service(request)

    # Shadow: send to new service, don't wait
    asyncio.create_task(new_service(request.copy()))  # Fire and forget

    return response  # User sees old service response
```

## Tool Ecosystem Quick Reference

| Tool Category | Options | When to Use |
|---------------|---------|-------------|
| **Feature Flags** | LaunchDarkly, Split.io, Flagsmith, Unleash | User-level targeting, instant rollback |
| **Canary/Blue-Green** | Istio, Linkerd, AWS App Mesh, Flagger | Service mesh, traffic shifting |
| **A/B Testing** | Optimizely, VWO, Google Optimize | Business metric validation |
| **Observability** | DataDog, New Relic, Honeycomb, Grafana | Metrics, traces, logs correlation |
| **Statistical Analysis** | Statsig, Eppo, GrowthBook | Automated significance testing |

**Recommendation for starting:** Feature flags (Flagsmith for self-hosted, LaunchDarkly for SaaS) + existing observability

## Your First Production Test

**Goal:** Safely test a small feature with feature flags

**Week 1: Setup**

1. **Choose feature flag tool**
   - Self-hosted: Flagsmith (free, open source)
   - SaaS: LaunchDarkly (free tier: 1000 MAU)

2. **Instrument code**
   ```python
   if feature_flags.enabled("my-first-test", user_id):
       return new_feature(user_id)
   else:
       return old_feature(user_id)
   ```

3. **Set up monitoring**
   - Error rate dashboard
   - Latency percentiles (p50, p95, p99)
   - Business metric (conversion, completion rate)

4. **Define rollback criteria**
   - Error rate > +5%
   - p99 latency > +50%
   - Business metric < -10%

**Week 2: Test Execution**

**Day 1-2:** Internal users (10 people)
- Enable flag for 10 employee user IDs
- Monitor for errors, gather feedback

**Day 3-5:** Canary (1% of users)
- Enable for 1% random sample
- Monitor metrics every hour
- Rollback if any threshold exceeded

**Day 6-8:** Small rollout (5%)
- If canary successful, increase to 5%
- Continue monitoring

**Day 9-14:** Full rollout (100%)
- Gradual increase: 25% → 50% → 100%
- Monitor for 7 days at 100%

**Week 3: Cleanup**

- Remove flag from code
- Archive flag in dashboard
- Document learnings

## Common Mistakes

### ❌ Expanding Rollout Too Fast
**Fix:** Follow minimum dwell times (24 hours per phase)

---

### ❌ Monitoring Only After Issues
**Fix:** Dashboard ready before first rollout, alerts configured

---

### ❌ No Rollback Practice
**Fix:** Test rollback in staging before production

---

### ❌ Ignoring Business Metrics
**Fix:** Technical metrics AND business metrics required for go/no-go decisions

## Quick Reference

**Technique Selection:**
- User-specific: Feature flags
- Deployment safety: Canary
- Performance comparison: Shadow traffic
- Business validation: A/B testing

**Blast Radius Progression:**
Internal → 1% → 5% → 25% → 50% → 100%

**Kill Switch Thresholds:**
- Error rate: > +5%
- p99 latency: > +50%
- Business metrics: > -10%

**Minimum Sample Sizes:**
- A/B test: 1000 samples per variant
- Canary: 24 hours observation

**Tool Recommendations:**
- Feature flags: LaunchDarkly, Flagsmith
- Canary: Istio, Flagger
- Observability: DataDog, Grafana

## Bottom Line

**Production testing is safe with three controls: exposure limits, observability, instant rollback.**

Start with feature flags, use progressive rollout (1% → 5% → 25% → 100%), monitor technical + business metrics, and always have a kill switch.
