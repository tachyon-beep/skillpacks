---
name: deployment-strategies
description: Master safe AI model deployment through A/B testing, canary releases, shadow mode, blue-green deployments, and automated rollback with statistical validation. Use when deploying models to production with traffic splitting and risk mitigation.
---

# Deployment Strategies for AI Models

## When to Use This Skill

Use this skill when:
- Deploying new AI models to production
- Comparing model versions in real traffic
- Gradually rolling out model updates
- Testing models without user impact (shadow mode)
- Building automated rollback procedures
- Validating model improvements with statistical rigor
- Managing feature flags for model control

**When NOT to use:** Development environments or single-user testing where gradual rollout isn't needed.

## Core Principle

**Instant deployment breaks production. Gradual deployment with validation saves production.**

Without safe deployment:
- Instant 100% deployment: One bad model breaks all users
- No A/B testing: Can't prove new model is better
- Canary without metrics: Deploy blindly, detect issues after damage
- Shadow mode forever: Never promote, wasted computation
- No rollback plan: Scramble to fix when things break

**Formula:** Shadow mode (validate without impact) → Canary 5% (detect issues early) → A/B test 50/50 (statistical proof) → Full rollout 100% (confident deployment) + Automated rollback (safety net) = Production-ready deployment.

## Deployment Framework

```
┌─────────────────────────────────────────┐
│      1. Shadow Mode (0% user impact)    │
│  Run new model alongside old, compare   │
│  Don't serve to users, collect metrics  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      2. Canary (5% → 25%)               │
│  Serve to small % of traffic            │
│  Monitor error rates, latency           │
│  Auto-rollback if metrics degrade       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      3. A/B Test (50% / 50%)            │
│  Split traffic evenly                   │
│  Measure statistical significance       │
│  Compare quality, latency, cost         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      4. Full Rollout (100%)             │
│  Promote winner to all traffic          │
│  Keep old model warm for rollback       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      5. Monitor & Rollback              │
│  Track metrics continuously             │
│  Automatic rollback on regression       │
└─────────────────────────────────────────┘
```

---

## RED Phase: What Goes Wrong (600-800 lines)

### Failure 1: Instant 100% Deployment

**Context:** Deploy new model directly to 100% of traffic, no gradual rollout.

**What Happens:**

```python
# BAD: Instant deployment to all users
def deploy_model_instant():
    """Deploy new model to 100% traffic immediately."""
    config.MODEL_VERSION = "v2.0"  # Everyone gets new model NOW
    restart_service()
    print("Model v2.0 deployed to 100% traffic!")
```

**User does this:**
```
User: "Deploy the new sentiment model v2.0 to production."

Agent: "Sure! I'll update the config and restart:

def deploy_new_model():
    update_config('MODEL_VERSION', 'v2.0')
    restart_service()

The new model is now live for all users!"
```

**Why This Fails:**

1. **No validation:** What if v2.0 has a bug?
2. **All users affected:** One bad model breaks everyone
3. **No comparison:** Can't measure if v2.0 is better than v1.0
4. **Hard to rollback:** Need emergency deployment to revert

**Real Production Impact:**
```
09:00 - Deploy v2.0 to 100%
09:05 - Error rate spikes from 0.1% → 15% (150× worse!)
09:10 - Customer support flooded with complaints
09:15 - Engineers scramble to diagnose
09:30 - Emergency rollback deployed
09:45 - Service stabilizes, but 45 minutes of downtime

Cost: 45 min downtime, 10,000 failed requests, angry customers
```

**What Agent Should Say:**

"STOP! Never deploy instantly to 100%. Use gradual rollout:

1. **Shadow mode first** (0% user impact): Run v2.0 alongside v1.0, compare outputs
2. **Canary 5%**: Serve v2.0 to 5% of users, monitor metrics
3. **Expand to 25%** if metrics good
4. **A/B test 50/50** to measure improvement
5. **Full rollout 100%** only after statistical validation

**Implementation:**
```python
# GOOD: Gradual deployment with validation
deployment_plan = {
    'shadow': {'duration': '24h', 'traffic': 0},
    'canary_5': {'duration': '2h', 'traffic': 5},
    'canary_25': {'duration': '6h', 'traffic': 25},
    'ab_test': {'duration': '48h', 'traffic': 50},
    'full': {'traffic': 100}
}

def deploy_gradual(model_version, stage):
    set_traffic_split(old=100-stage['traffic'], new=stage['traffic'])
    monitor_metrics(duration=stage['duration'])
    if metrics_acceptable():
        proceed_to_next_stage()
    else:
        rollback()
```

**Principle: Deploy gradually. Shadow → Canary → A/B → Full. Validate at each stage.**"

---

### Failure 2: A/B Test Without Statistical Validation

**Context:** Run A/B test but don't measure statistical significance, make decision on small sample.

**What Happens:**

```python
# BAD: No statistical testing
def compare_models():
    """Compare models without statistical rigor."""
    model_a_accuracy = 0.87
    model_b_accuracy = 0.89

    if model_b_accuracy > model_a_accuracy:
        print("Model B is better! Deploy it.")
        deploy(model_b)
```

**User does this:**
```
User: "Run A/B test for 1 hour. Model A got 87% accuracy, Model B got 89%. Deploy B."

Agent: "Model B has higher accuracy (89% vs 87%), so it's better. Deploying Model B:

def deploy_winner():
    if model_b_score > model_a_score:
        deploy(model_b)

Model B is now live!"
```

**Why This Fails:**

1. **No sample size calculation:** 1 hour might be 100 requests (too small)
2. **No confidence interval:** Is 89% vs 87% statistically significant?
3. **No variance consideration:** Model B might be 89% ± 5% (could be 84-94%)
4. **Could be random noise:** 2% difference might be luck

**Real Production Impact:**
```
A/B test for 1 hour:
- Model A: 87% on 50 samples (43 correct, 7 wrong)
- Model B: 89% on 50 samples (44 correct, 6 wrong)
- Difference: 1 more correct prediction

Deploy Model B to 100%

After 1 week:
- Model B: 85% on 10,000 samples (actually worse!)
- Model A was better, but sample size was too small to tell

Cost: Deployed worse model, degraded user experience, need to rollback
```

**Statistical Reality:**
```python
# Calculate statistical significance
from scipy.stats import proportions_ztest

# Model A: 43/50 = 86%, Model B: 44/50 = 88%
stat, p_value = proportions_ztest([43, 44], [50, 50])
print(f"p-value: {p_value:.3f}")  # p=0.72 (NOT significant!)

# Need p < 0.05 for significance
# With only 50 samples each, difference is random noise
```

**What Agent Should Say:**

"STOP! You can't conclude Model B is better from 100 samples.

**Problem 1: No sample size calculation**
```python
# Required sample size for 80% power, 5% significance, 2% effect size
from statsmodels.stats.power import zt_ind_solve_power

required_n = zt_ind_solve_power(
    effect_size=0.02 / 0.1,  # 2% improvement, 10% baseline std
    alpha=0.05,  # 5% significance level
    power=0.8,  # 80% power
)
print(f"Need {required_n:.0f} samples per model")  # ~3,100 samples!

# You only have 50 samples per model - need 62× more data
```

**Problem 2: No significance testing**
```python
# Proper statistical test
def ab_test_significance(a_correct, a_total, b_correct, b_total):
    from scipy.stats import proportions_ztest

    stat, p_value = proportions_ztest(
        [a_correct, b_correct],
        [a_total, b_total]
    )

    if p_value < 0.05:
        return "Statistically significant"
    else:
        return f"NOT significant (p={p_value:.3f}). Need more data."

result = ab_test_significance(43, 50, 44, 50)
print(result)  # "NOT significant (p=0.72). Need more data."
```

**Problem 3: No confidence intervals**
```python
# Calculate confidence intervals
from statsmodels.stats.proportion import proportion_confint

ci_a = proportion_confint(43, 50, alpha=0.05, method='wilson')
ci_b = proportion_confint(44, 50, alpha=0.05, method='wilson')

print(f"Model A: 86% CI=[{ci_a[0]:.1%}, {ci_a[1]:.1%}]")  # [73%, 94%]
print(f"Model B: 88% CI=[{ci_b[0]:.1%}, {ci_b[1]:.1%}]")  # [76%, 95%]

# Confidence intervals overlap heavily! Can't distinguish models.
```

**Fix: Proper A/B testing**
```python
def run_ab_test_properly(model_a, model_b, min_samples=3000):
    # Calculate required sample size
    required_n = calculate_sample_size(
        effect_size=0.02,  # Minimum detectable difference
        alpha=0.05,  # Significance level
        power=0.8  # Statistical power
    )

    # Collect data
    results_a = []
    results_b = []

    while len(results_a) < max(required_n, min_samples):
        # 50/50 traffic split
        result = route_request(a_weight=0.5, b_weight=0.5)
        results_a.append(result['a'])
        results_b.append(result['b'])

    # Statistical test
    stat, p_value = proportions_ztest(
        [sum(results_a), sum(results_b)],
        [len(results_a), len(results_b)]
    )

    # Calculate effect size
    effect_size = (sum(results_b)/len(results_b) -
                   sum(results_a)/len(results_a))

    # Confidence intervals
    ci_a = proportion_confint(sum(results_a), len(results_a))
    ci_b = proportion_confint(sum(results_b), len(results_b))

    # Decision
    if p_value < 0.05 and effect_size > 0.01:  # Significant AND meaningful
        return {
            'winner': 'Model B',
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence': 'HIGH',
            'recommendation': 'Deploy Model B'
        }
    else:
        return {
            'winner': 'INCONCLUSIVE',
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence': 'LOW',
            'recommendation': 'Keep Model A or collect more data'
        }
```

**Principle: A/B tests need statistical rigor. Calculate sample size, test significance, check confidence intervals. Don't trust small samples.**"

---

### Failure 3: Canary Deployment Without Metrics

**Context:** Deploy to 5% of users but don't monitor metrics, just "see if anyone complains."

**What Happens:**

```python
# BAD: Canary without monitoring
def canary_deploy():
    """Deploy to 5% with no monitoring."""
    set_traffic_split(old=95, new=5)
    print("Deployed to 5% of users. Let's wait and see...")
    time.sleep(3600)  # Wait 1 hour, no monitoring
    print("No complaints? Deploy to 100%!")
    set_traffic_split(old=0, new=100)
```

**User does this:**
```
User: "Start canary deployment at 5% for the new model."

Agent: "Done! I've set 5% of traffic to the new model:

def canary():
    route_traffic(old=95, new=5)
    wait(hours=2)
    if no_complaints():
        deploy_full()

The canary is running. I'll check in 2 hours."
```

**Why This Fails:**

1. **No automated metrics:** Relying on users to complain
2. **Users don't always complain:** They silently leave
3. **5% = small sample:** Might miss rare errors
4. **No rollback trigger:** Just "wait and see"
5. **Silent failures:** API errors, slow responses unnoticed

**Real Production Impact:**
```
10:00 - Canary deploy to 5% (5,000 users/hour)
10:00-12:00 - No complaints received
12:00 - Deploy to 100% based on "no news is good news"
12:30 - Users report slow responses (2s → 8s latency)
13:00 - Error rate 10× higher (0.1% → 1%)
13:30 - Emergency rollback

Post-mortem:
- Canary phase HAD issues (8s latency for 5% of users)
- Users didn't complain (they just refreshed)
- No automated monitoring caught the problem
- Full deployment amplified issue to 100% of users

Cost: 1.5 hours of degraded service, lost user trust
```

**What metrics should have caught it:**
```python
# Canary metrics that would have caught the issue
canary_metrics = {
    'latency_p50': 8.2,  # vs 2.0s baseline (4× WORSE) ← RED FLAG
    'latency_p95': 15.3,  # vs 4.0s baseline (4× WORSE) ← RED FLAG
    'error_rate': 1.2,  # vs 0.1% baseline (12× WORSE) ← RED FLAG
    'timeout_rate': 2.1,  # vs 0.5% baseline (4× WORSE) ← RED FLAG
}

# Automated check would have rolled back immediately
```

**What Agent Should Say:**

"STOP! Never run canary without automated metrics monitoring.

**Problem: No observability**
```
Current: Deploy 5%, wait, 'no complaints' = success
Reality: Users don't complain, they leave
```

**Fix: Canary with automated monitoring**
```python
def canary_deploy_with_monitoring(
    new_model,
    canary_percent=5,
    duration_minutes=120,
    check_interval_minutes=5
):
    \"\"\"
    Canary deployment with automated metrics monitoring.

    Monitors:
    - Latency (p50, p95, p99)
    - Error rate
    - Timeout rate
    - User satisfaction (if available)

    Auto-rollback if any metric degrades beyond threshold.
    \"\"\"

    # Baseline metrics from old model
    baseline = get_baseline_metrics(hours=24)

    # Start canary
    set_traffic_split(old=100-canary_percent, new=canary_percent)
    print(f"Canary started: {canary_percent}% traffic to new model")

    # Monitor for duration
    for elapsed in range(0, duration_minutes, check_interval_minutes):
        # Get canary metrics
        canary_metrics = get_canary_metrics(minutes=check_interval_minutes)

        # Compare to baseline
        checks = {
            'latency_p50': canary_metrics['latency_p50'] < baseline['latency_p50'] * 1.2,  # Allow 20% increase
            'latency_p95': canary_metrics['latency_p95'] < baseline['latency_p95'] * 1.5,  # Allow 50% increase
            'error_rate': canary_metrics['error_rate'] < baseline['error_rate'] * 2.0,  # Allow 2× increase
            'timeout_rate': canary_metrics['timeout_rate'] < baseline['timeout_rate'] * 2.0,  # Allow 2× increase
        }

        # Check for failures
        failed_checks = [k for k, v in checks.items() if not v]

        if failed_checks:
            print(f"ROLLBACK TRIGGERED! Failed checks: {failed_checks}")
            print(f"Baseline: {baseline}")
            print(f"Canary:   {canary_metrics}")

            # Automatic rollback
            set_traffic_split(old=100, new=0)
            alert_team(
                message=f"Canary rollback: {failed_checks}",
                metrics={'baseline': baseline, 'canary': canary_metrics}
            )
            return {'status': 'ROLLED_BACK', 'reason': failed_checks}

        print(f"Canary healthy at {elapsed + check_interval_minutes} min: {canary_metrics}")
        time.sleep(check_interval_minutes * 60)

    # Canary succeeded
    print("Canary succeeded! Ready to expand.")
    return {'status': 'SUCCESS', 'metrics': canary_metrics}

# Usage
result = canary_deploy_with_monitoring(
    new_model='v2.0',
    canary_percent=5,
    duration_minutes=120
)

if result['status'] == 'SUCCESS':
    # Expand to 25%
    canary_deploy_with_monitoring(new_model='v2.0', canary_percent=25)
```

**Key metrics to monitor:**

| Metric | Threshold | Why |
|--------|-----------|-----|
| Latency p50 | < 1.2× baseline | Most users shouldn't notice slowdown |
| Latency p95 | < 1.5× baseline | Tail latency matters for UX |
| Error rate | < 2× baseline | Small increase acceptable, large = bug |
| Timeout rate | < 2× baseline | Timeouts = bad UX |
| CPU usage | < 1.5× baseline | Cost and capacity concern |
| Memory usage | < 1.5× baseline | Memory leaks show up here |

**Principle: Canary without metrics is hope, not engineering. Monitor automatically, rollback automatically.**"

---

### Failure 4: Shadow Mode Without Comparison

**Context:** Run new model in shadow mode but never compare outputs to old model.

**What Happens:**

```python
# BAD: Shadow mode with no comparison
def shadow_mode():
    """Run new model in background, don't compare."""
    old_result = model_v1(input)
    new_result = model_v2(input)  # Compute but ignore

    return old_result  # Always return old model

# New model runs forever in shadow, never evaluated
```

**User does this:**
```
User: "Run the new model in shadow mode alongside the current model."

Agent: "I'll set up shadow mode:

def shadow_deploy():
    # Serve old model to users
    result = old_model.predict(input)

    # Run new model in background (don't serve)
    new_model.predict(input)  # Async, don't wait

    return result

Shadow mode active! The new model is running but not served to users."
```

**Why This Fails:**

1. **No comparison:** Running two models but not comparing outputs
2. **Wasted compute:** New model runs but insights ignored
3. **No decision criteria:** When to promote new model?
4. **Stays in shadow forever:** No graduation path

**Real Production Impact:**
```
Week 1: Shadow mode running, new model processes 1M requests
Week 2: Still in shadow mode, 2M requests total
Week 3: Engineer asks: "Should we promote the new model?"
Answer: "No idea. We didn't compare outputs."

Cost: 3 weeks of wasted compute (running but not learning)
```

**What Agent Should Say:**

"Shadow mode without comparison is wasted compute. You need to COMPARE outputs and decide promotion criteria.

**Fix: Shadow mode with comparison and decision criteria**
```python
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ShadowResult:
    input: str
    old_output: str
    new_output: str
    old_latency: float
    new_latency: float
    match: bool
    timestamp: float

class ShadowModeComparison:
    \"\"\"
    Run new model in shadow, compare to old model, decide when to promote.
    \"\"\"

    def __init__(self, old_model, new_model, sample_rate=1.0):
        self.old_model = old_model
        self.new_model = new_model
        self.sample_rate = sample_rate
        self.results: List[ShadowResult] = []

    def predict_with_shadow(self, input: str) -> str:
        \"\"\"
        Predict with old model, run new model in shadow for comparison.
        \"\"\"
        import time

        # Old model (served to users)
        start = time.time()
        old_output = self.old_model.predict(input)
        old_latency = time.time() - start

        # New model (shadow, not served)
        if np.random.random() < self.sample_rate:
            start = time.time()
            new_output = self.new_model.predict(input)
            new_latency = time.time() - start

            # Compare outputs
            match = self._compare_outputs(old_output, new_output)

            # Store for analysis
            self.results.append(ShadowResult(
                input=input,
                old_output=old_output,
                new_output=new_output,
                old_latency=old_latency,
                new_latency=new_latency,
                match=match,
                timestamp=time.time()
            ))

        return old_output  # Always serve old model

    def _compare_outputs(self, old: str, new: str) -> bool:
        \"\"\"Compare outputs (exact match or semantic similarity).\"\"\"
        # For classification: exact match
        if old in ['positive', 'negative', 'neutral']:
            return old == new

        # For text generation: semantic similarity
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        old_emb = model.encode(old)
        new_emb = model.encode(new)

        similarity = np.dot(old_emb, new_emb) / (
            np.linalg.norm(old_emb) * np.linalg.norm(new_emb)
        )

        return similarity > 0.9  # 90% similar = match

    def get_analysis(self) -> Dict:
        \"\"\"
        Analyze shadow mode results and recommend promotion.
        \"\"\"
        if len(self.results) < 100:
            return {
                'status': 'INSUFFICIENT_DATA',
                'message': f'Only {len(self.results)} samples. Need 100+ for decision.',
                'recommendation': 'Continue shadow mode'
            }

        # Calculate metrics
        agreement_rate = np.mean([r.match for r in self.results])

        old_latency_p50 = np.median([r.old_latency for r in self.results])
        new_latency_p50 = np.median([r.new_latency for r in self.results])

        old_latency_p95 = np.percentile([r.old_latency for r in self.results], 95)
        new_latency_p95 = np.percentile([r.new_latency for r in self.results], 95)

        # Decision criteria
        latency_acceptable = new_latency_p95 < old_latency_p95 * 1.5  # Max 50% slower
        agreement_acceptable = agreement_rate > 0.85  # 85% agreement

        # Recommendation
        if latency_acceptable and agreement_acceptable:
            recommendation = 'PROMOTE_TO_CANARY'
            message = (
                f'Shadow mode successful! '
                f'Agreement: {agreement_rate:.1%}, '
                f'Latency p95: {new_latency_p95:.3f}s vs {old_latency_p95:.3f}s'
            )
        elif not latency_acceptable:
            recommendation = 'OPTIMIZE_LATENCY'
            message = (
                f'New model too slow: '
                f'{new_latency_p95:.3f}s vs {old_latency_p95:.3f}s (>{1.5:.1f}× threshold)'
            )
        else:  # not agreement_acceptable
            recommendation = 'INVESTIGATE_DISAGREEMENT'
            message = (
                f'Low agreement: {agreement_rate:.1%}. '
                f'Review disagreement cases before promoting.'
            )

        return {
            'status': 'ANALYSIS_COMPLETE',
            'samples': len(self.results),
            'agreement_rate': agreement_rate,
            'old_latency_p50': old_latency_p50,
            'new_latency_p50': new_latency_p50,
            'old_latency_p95': old_latency_p95,
            'new_latency_p95': new_latency_p95,
            'recommendation': recommendation,
            'message': message
        }

# Usage
shadow = ShadowModeComparison(old_model, new_model, sample_rate=0.1)

# Run for 1 week
for request in production_traffic:
    result = shadow.predict_with_shadow(request)
    serve_to_user(result)

# Analyze after 1 week
analysis = shadow.get_analysis()
print(analysis)

# Example output:
{
    'status': 'ANALYSIS_COMPLETE',
    'samples': 10000,
    'agreement_rate': 0.92,  # 92% agree
    'old_latency_p50': 0.15,
    'new_latency_p50': 0.18,  # 20% slower (acceptable)
    'old_latency_p95': 0.45,
    'new_latency_p95': 0.58,  # 29% slower (acceptable < 50%)
    'recommendation': 'PROMOTE_TO_CANARY',
    'message': 'Shadow mode successful! Agreement: 92%, Latency p95: 0.58s vs 0.45s'
}
```

**Principle: Shadow mode is for comparison, not just running. Compare outputs, latency, errors. Decide promotion criteria upfront.**"

---

### Failure 5: No Rollback Plan

**Context:** Deploy new model but have no rollback procedure when things go wrong.

**What Happens:**

```python
# BAD: No rollback plan
def deploy():
    """Deploy with no rollback."""
    config.MODEL = 'v2.0'
    restart_service()
    # Hope nothing breaks!
```

**User does this:**
```
User: "Deploy model v2.0."

Agent: "Deployed!

def deploy():
    set_model('v2.0')
    restart()

Model v2.0 is live."

[30 minutes later]

User: "Error rate is spiking! Roll back to v1.0!"

Agent: "Um... let me see... I need to find the v1.0 model file... where is it? And I need to update config... and restart... this will take a few minutes..."

[10 minutes of scrambling]

Agent: "OK, rolled back. That took 40 minutes of downtime total."
```

**Why This Fails:**

1. **No automated rollback:** Manual process takes time
2. **Old model not warm:** Need to load and warm up
3. **Config changes manual:** Error-prone under pressure
4. **No rollback testing:** Don't know if rollback works until emergency
5. **Delayed response:** 40 minutes vs 30 seconds with automation

**Real Production Impact:**
```
10:00 - Deploy v2.0 to 100%
10:30 - Error rate spikes, need rollback
10:30 - Begin manual rollback
  10:32 - Find v1.0 model artifact (2 min)
  10:35 - Update config (3 min)
  10:38 - Restart service (3 min)
  10:40 - Wait for healthcheck (2 min)
10:40 - Service restored (10 minutes downtime)

Better with automation:
10:30 - Error rate spikes, trigger automatic rollback
10:30:30 - Rollback complete (30 seconds)
```

**What Agent Should Say:**

"NEVER deploy without automated rollback. Build rollback first, then deploy.

**Fix: Automated rollback with blue-green deployment**
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time

class ModelVersion(Enum):
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentState:
    active_version: ModelVersion
    blue_model: str
    green_model: str
    blue_weight: int
    green_weight: int

class BlueGreenDeployment:
    \"\"\"
    Blue-green deployment with instant rollback.

    Strategy:
    - Blue = current production model
    - Green = new model being deployed
    - Traffic splits between blue and green
    - Rollback = instant traffic shift to blue
    \"\"\"

    def __init__(self, blue_model: str):
        self.state = DeploymentState(
            active_version=ModelVersion.BLUE,
            blue_model=blue_model,
            green_model=None,
            blue_weight=100,
            green_weight=0
        )

        # Keep both models warm
        self.models = {
            ModelVersion.BLUE: load_model(blue_model),
            ModelVersion.GREEN: None
        }

    def deploy_green(self, green_model: str):
        \"\"\"Deploy new model to green slot.\"\"\"
        print(f"Loading green model: {green_model}")
        self.models[ModelVersion.GREEN] = load_model(green_model)
        self.state.green_model = green_model
        print("Green model loaded and warm")

    def shift_traffic(self, blue_weight: int, green_weight: int):
        \"\"\"Shift traffic between blue and green.\"\"\"
        if blue_weight + green_weight != 100:
            raise ValueError("Weights must sum to 100")

        self.state.blue_weight = blue_weight
        self.state.green_weight = green_weight

        # Update load balancer
        update_load_balancer({
            'blue': blue_weight,
            'green': green_weight
        })

        print(f"Traffic split: Blue={blue_weight}%, Green={green_weight}%")

    def rollback(self, reason: str = "Manual rollback"):
        \"\"\"
        INSTANT rollback to blue (stable version).

        Takes ~1 second (just update load balancer).
        \"\"\"
        print(f"ROLLBACK TRIGGERED: {reason}")
        print(f"Shifting 100% traffic to Blue ({self.state.blue_model})")

        self.shift_traffic(blue_weight=100, green_weight=0)

        alert_team(
            message=f"Rollback executed: {reason}",
            old_state={'blue': self.state.blue_weight, 'green': self.state.green_weight},
            new_state={'blue': 100, 'green': 0}
        )

        print("Rollback complete (< 1 second)")

    def promote_green(self):
        \"\"\"
        Promote green to blue (make green the new stable).

        Process:
        1. Green is at 100% traffic (already tested)
        2. Swap blue ↔ green labels
        3. Old blue becomes new green (ready for next deployment)
        \"\"\"
        print("Promoting green to blue")

        # Swap models
        old_blue = self.state.blue_model
        old_blue_model = self.models[ModelVersion.BLUE]

        self.state.blue_model = self.state.green_model
        self.state.green_model = old_blue

        self.models[ModelVersion.BLUE] = self.models[ModelVersion.GREEN]
        self.models[ModelVersion.GREEN] = old_blue_model

        # Update traffic (blue=100%, green=0%)
        self.state.blue_weight = 100
        self.state.green_weight = 0

        print(f"Promotion complete: {self.state.blue_model} is now stable")

    def gradual_rollout(
        self,
        green_model: str,
        stages: list = [5, 25, 50, 100],
        stage_duration_minutes: int = 60
    ):
        \"\"\"
        Gradual rollout with automatic rollback on errors.
        \"\"\"
        # Deploy to green slot
        self.deploy_green(green_model)

        # Monitor metrics
        baseline_metrics = get_metrics(window_minutes=60)

        for stage in stages:
            print(f"\\n=== Stage: {stage}% to green ===")

            # Shift traffic
            self.shift_traffic(blue_weight=100-stage, green_weight=stage)

            # Monitor for duration
            print(f"Monitoring for {stage_duration_minutes} minutes...")

            for minute in range(stage_duration_minutes):
                time.sleep(60)

                # Check metrics every minute
                current_metrics = get_metrics(window_minutes=5)

                # Automated health check
                health = self._check_health(baseline_metrics, current_metrics)

                if not health['healthy']:
                    print(f"Health check FAILED: {health['reason']}")
                    self.rollback(reason=health['reason'])
                    return {'status': 'ROLLED_BACK', 'reason': health['reason']}

                if (minute + 1) % 10 == 0:
                    print(f"  {minute + 1}/{stage_duration_minutes} min - Healthy")

            print(f"Stage {stage}% complete. Metrics healthy.")

        # All stages passed, promote green to blue
        self.promote_green()

        return {'status': 'SUCCESS', 'model': green_model}

    def _check_health(self, baseline: dict, current: dict) -> dict:
        \"\"\"Check if current metrics are healthy compared to baseline.\"\"\"
        checks = {
            'error_rate': current['error_rate'] < baseline['error_rate'] * 2.0,
            'latency_p95': current['latency_p95'] < baseline['latency_p95'] * 1.5,
            'timeout_rate': current['timeout_rate'] < baseline['timeout_rate'] * 2.0,
        }

        failed = [k for k, v in checks.items() if not v]

        if failed:
            return {
                'healthy': False,
                'reason': f"Metrics degraded: {failed}. Current: {current}, Baseline: {baseline}"
            }

        return {'healthy': True}

# Usage
deployment = BlueGreenDeployment(blue_model='v1.0')

# Deploy v2.0 with gradual rollout and automatic rollback
result = deployment.gradual_rollout(
    green_model='v2.0',
    stages=[5, 25, 50, 100],  # Canary 5% → 25% → A/B 50% → Full 100%
    stage_duration_minutes=60
)

# If any stage fails, automatic rollback to v1.0 (< 1 second)
# If all stages pass, v2.0 promoted to stable

print(result)
# {'status': 'SUCCESS', 'model': 'v2.0'}
# or
# {'status': 'ROLLED_BACK', 'reason': 'Metrics degraded: error_rate. Current: {...}, Baseline: {...}'}
```

**Rollback timing comparison:**

| Method | Rollback Time | Risk |
|--------|---------------|------|
| Manual | 5-10 minutes | High (human error, stress) |
| Scripted | 2-3 minutes | Medium (still manual trigger) |
| Automated | < 30 seconds | Low (instant, no human) |
| Blue-green | < 1 second | Minimal (just traffic shift) |

**Principle: Build rollback before deploying. Automated, instant, tested. Blue-green deployment makes rollback a config change, not a deploy.**"

---

## Summary of RED Phase Failures

**5 Failures Covered:**

1. **Instant 100% deployment** → All users impacted by bugs
2. **A/B test without statistics** → Wrong conclusions from small samples
3. **Canary without metrics** → Silent failures go unnoticed
4. **Shadow mode without comparison** → Wasted compute, no learning
5. **No rollback plan** → Slow recovery from failures

**Common themes:**
- **No validation** → Hope-driven deployment
- **No automation** → Manual processes fail under pressure
- **No metrics** → Flying blind
- **No gradual rollout** → All-or-nothing risk
- **No rollback** → Long recovery time

**Core insight:** Safe deployment requires automation, metrics, gradual rollout, and instant rollback. Each step must validate before proceeding.

---

## GREEN Phase: Safe Deployment Patterns (900-1200 lines)

### Pattern 1: A/B Testing with Statistical Validation

**Goal:** Compare two models with statistical rigor to make confident decisions.

**Complete Implementation:**

```python
import numpy as np
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_confint
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time

class ABTestStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    INCONCLUSIVE = "inconclusive"
    A_WINS = "a_wins"
    B_WINS = "b_wins"
    TIE = "tie"

@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    min_sample_size: int = 1000  # Minimum samples per variant
    significance_level: float = 0.05  # Alpha (5% significance)
    power: float = 0.8  # 80% statistical power
    min_effect_size: float = 0.02  # Minimum detectable effect (2%)
    traffic_split: float = 0.5  # 50/50 split

@dataclass
class ABTestResult:
    """Result of A/B test."""
    status: ABTestStatus
    winner: Optional[str]
    p_value: float
    effect_size: float
    confidence_interval_a: Tuple[float, float]
    confidence_interval_b: Tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    metric_a: float
    metric_b: float
    required_sample_size: int
    recommendation: str

class ABTest:
    """
    A/B testing framework with statistical validation.

    Features:
    - Sample size calculation (power analysis)
    - Statistical significance testing (z-test)
    - Confidence intervals
    - Effect size calculation
    - Multi-metric evaluation
    - Automatic decision making
    """

    def __init__(self, model_a, model_b, config: ABTestConfig = None):
        self.model_a = model_a
        self.model_b = model_b
        self.config = config or ABTestConfig()

        self.results_a = []
        self.results_b = []
        self.metadata_a = []
        self.metadata_b = []

    def calculate_required_sample_size(
        self,
        baseline_rate: float = 0.5,
        effect_size: float = None
    ) -> int:
        """
        Calculate required sample size for statistical power.

        Args:
            baseline_rate: Expected baseline conversion/success rate
            effect_size: Minimum detectable effect (default from config)

        Returns:
            Required sample size per variant
        """
        effect_size = effect_size or self.config.min_effect_size

        # Convert effect size to Cohen's h
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        cohens_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Calculate required sample size
        n = zt_ind_solve_power(
            effect_size=cohens_h,
            alpha=self.config.significance_level,
            power=self.config.power,
            ratio=1.0,  # Equal sample sizes
            alternative='two-sided'
        )

        return int(np.ceil(n))

    def route_request(self, request) -> Tuple[str, any]:
        """
        Route request to A or B based on traffic split.

        Returns:
            (variant, result) where variant is 'a' or 'b'
        """
        if np.random.random() < self.config.traffic_split:
            variant = 'a'
            result = self.model_a.predict(request)
        else:
            variant = 'b'
            result = self.model_b.predict(request)

        return variant, result

    def record_result(self, variant: str, success: bool, metadata: dict = None):
        """
        Record result for variant.

        Args:
            variant: 'a' or 'b'
            success: Whether the prediction was successful (1) or not (0)
            metadata: Optional metadata (latency, user_id, etc.)
        """
        if variant == 'a':
            self.results_a.append(1 if success else 0)
            self.metadata_a.append(metadata or {})
        else:
            self.results_b.append(1 if success else 0)
            self.metadata_b.append(metadata or {})

    def test_significance(self) -> ABTestResult:
        """
        Test statistical significance of results.

        Returns:
            ABTestResult with decision and metrics
        """
        n_a = len(self.results_a)
        n_b = len(self.results_b)

        # Check minimum sample size
        required_n = self.calculate_required_sample_size()

        if n_a < required_n or n_b < required_n:
            return ABTestResult(
                status=ABTestStatus.INCONCLUSIVE,
                winner=None,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval_a=(0.0, 0.0),
                confidence_interval_b=(0.0, 0.0),
                sample_size_a=n_a,
                sample_size_b=n_b,
                metric_a=0.0,
                metric_b=0.0,
                required_sample_size=required_n,
                recommendation=f"Continue test. Need {required_n - min(n_a, n_b)} more samples."
            )

        # Calculate metrics
        successes_a = sum(self.results_a)
        successes_b = sum(self.results_b)

        rate_a = successes_a / n_a
        rate_b = successes_b / n_b

        # Statistical test (two-proportion z-test)
        from statsmodels.stats.proportion import proportions_ztest

        stat, p_value = proportions_ztest(
            [successes_a, successes_b],
            [n_a, n_b]
        )

        # Confidence intervals
        ci_a = proportion_confint(successes_a, n_a, alpha=self.config.significance_level, method='wilson')
        ci_b = proportion_confint(successes_b, n_b, alpha=self.config.significance_level, method='wilson')

        # Effect size
        effect_size = rate_b - rate_a

        # Decision
        is_significant = p_value < self.config.significance_level
        is_meaningful = abs(effect_size) >= self.config.min_effect_size

        if is_significant and is_meaningful:
            if effect_size > 0:
                status = ABTestStatus.B_WINS
                winner = 'b'
                recommendation = f"Deploy Model B. {rate_b:.1%} vs {rate_a:.1%} (p={p_value:.4f})"
            else:
                status = ABTestStatus.A_WINS
                winner = 'a'
                recommendation = f"Keep Model A. {rate_a:.1%} vs {rate_b:.1%} (p={p_value:.4f})"
        elif is_significant and not is_meaningful:
            status = ABTestStatus.TIE
            winner = None
            recommendation = f"Models equivalent. Effect size {effect_size:.1%} below threshold {self.config.min_effect_size:.1%}."
        else:
            status = ABTestStatus.INCONCLUSIVE
            winner = None
            recommendation = f"No significant difference (p={p_value:.4f}). Consider longer test or accept tie."

        return ABTestResult(
            status=status,
            winner=winner,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval_a=ci_a,
            confidence_interval_b=ci_b,
            sample_size_a=n_a,
            sample_size_b=n_b,
            metric_a=rate_a,
            metric_b=rate_b,
            required_sample_size=required_n,
            recommendation=recommendation
        )

    def run_test(
        self,
        traffic_generator,
        max_duration_hours: int = 48,
        check_interval_minutes: int = 60
    ) -> ABTestResult:
        """
        Run A/B test with automatic stopping.

        Args:
            traffic_generator: Generator yielding (request, ground_truth) tuples
            max_duration_hours: Maximum test duration
            check_interval_minutes: How often to check for significance

        Returns:
            ABTestResult with final decision
        """
        start_time = time.time()
        last_check = start_time

        print(f"Starting A/B test: Model A vs Model B")
        print(f"Config: {self.config}")
        print(f"Required sample size: {self.calculate_required_sample_size()} per variant")

        for request, ground_truth in traffic_generator:
            # Route request
            variant, prediction = self.route_request(request)

            # Evaluate
            success = self._evaluate(prediction, ground_truth)

            # Record with metadata
            metadata = {
                'timestamp': time.time(),
                'request': request,
                'prediction': prediction,
                'ground_truth': ground_truth
            }
            self.record_result(variant, success, metadata)

            # Check for significance periodically
            if time.time() - last_check > check_interval_minutes * 60:
                result = self.test_significance()

                print(f"\n=== Check at {len(self.results_a) + len(self.results_b)} samples ===")
                print(f"Model A: {result.metric_a:.1%} ({result.sample_size_a} samples)")
                print(f"Model B: {result.metric_b:.1%} ({result.sample_size_b} samples)")
                print(f"Status: {result.status.value}")
                print(f"p-value: {result.p_value:.4f}")
                print(f"Effect size: {result.effect_size:+.1%}")
                print(f"Recommendation: {result.recommendation}")

                # Stop if conclusive
                if result.status in [ABTestStatus.A_WINS, ABTestStatus.B_WINS, ABTestStatus.TIE]:
                    print(f"\nTest concluded: {result.status.value}")
                    return result

                last_check = time.time()

            # Stop if max duration reached
            if time.time() - start_time > max_duration_hours * 3600:
                print(f"\nMax duration ({max_duration_hours}h) reached")
                result = self.test_significance()
                return result

        # Test ended (traffic exhausted)
        return self.test_significance()

    def _evaluate(self, prediction, ground_truth) -> bool:
        """Evaluate if prediction matches ground truth."""
        return prediction == ground_truth

    def analyze_segments(self, segment_key: str = 'user_type') -> Dict[str, ABTestResult]:
        """
        Analyze results by segments (e.g., user type, geography).

        Args:
            segment_key: Key in metadata to segment by

        Returns:
            Dict mapping segment to ABTestResult
        """
        # Group by segment
        segments_a = {}
        segments_b = {}

        for result, metadata in zip(self.results_a, self.metadata_a):
            segment = metadata.get(segment_key, 'unknown')
            if segment not in segments_a:
                segments_a[segment] = []
            segments_a[segment].append(result)

        for result, metadata in zip(self.results_b, self.metadata_b):
            segment = metadata.get(segment_key, 'unknown')
            if segment not in segments_b:
                segments_b[segment] = []
            segments_b[segment].append(result)

        # Analyze each segment
        segment_results = {}

        for segment in set(segments_a.keys()) | set(segments_b.keys()):
            results_a = segments_a.get(segment, [])
            results_b = segments_b.get(segment, [])

            # Create temporary AB test for segment
            segment_test = ABTest(self.model_a, self.model_b, self.config)
            segment_test.results_a = results_a
            segment_test.results_b = results_b

            segment_results[segment] = segment_test.test_significance()

        return segment_results


# Example usage
if __name__ == "__main__":
    # Mock models
    class ModelA:
        def predict(self, x):
            return "positive" if np.random.random() < 0.75 else "negative"

    class ModelB:
        def predict(self, x):
            return "positive" if np.random.random() < 0.78 else "negative"  # 3% better

    # Traffic generator (mock)
    def traffic_generator():
        for i in range(10000):
            request = f"Review {i}"
            ground_truth = "positive" if np.random.random() < 0.75 else "negative"
            yield request, ground_truth

    # Run A/B test
    ab_test = ABTest(ModelA(), ModelB())

    result = ab_test.run_test(
        traffic_generator(),
        max_duration_hours=48,
        check_interval_minutes=60
    )

    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Status: {result.status.value}")
    print(f"Winner: {result.winner}")
    print(f"Model A: {result.metric_a:.1%} CI=[{result.confidence_interval_a[0]:.1%}, {result.confidence_interval_a[1]:.1%}]")
    print(f"Model B: {result.metric_b:.1%} CI=[{result.confidence_interval_b[0]:.1%}, {result.confidence_interval_b[1]:.1%}]")
    print(f"Effect size: {result.effect_size:+.1%}")
    print(f"p-value: {result.p_value:.4f}")
    print(f"Recommendation: {result.recommendation}")
```

**Key Features:**

1. **Sample size calculation:** Power analysis ensures sufficient data
2. **Statistical testing:** Two-proportion z-test with significance level
3. **Confidence intervals:** Quantify uncertainty
4. **Effect size:** Measure practical significance
5. **Automatic stopping:** Stop when conclusive or time limit reached
6. **Segment analysis:** Analyze by user type, geography, etc.

**Usage guidelines:**

| Scenario | Min Sample Size | Duration | Traffic Split |
|----------|-----------------|----------|---------------|
| Small effect (2%) | 3,000/variant | 1-2 weeks | 50/50 |
| Medium effect (5%) | 500/variant | 3-5 days | 50/50 |
| Large effect (10%) | 200/variant | 1-2 days | 50/50 |

---

### Pattern 2: Canary Deployment with Automated Rollback

**Goal:** Gradually increase traffic to new model while monitoring metrics and auto-rollback on regression.

**Complete Implementation:**

```python
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanaryStage(Enum):
    SHADOW = "shadow"  # 0% user traffic
    CANARY_5 = "canary_5"  # 5% traffic
    CANARY_25 = "canary_25"  # 25% traffic
    AB_TEST = "ab_test"  # 50% traffic
    FULL = "full"  # 100% traffic

class CanaryStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ROLLED_BACK = "rolled_back"

@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    stages: List[Dict] = None  # List of {'stage': CanaryStage, 'duration_minutes': int}
    check_interval_minutes: int = 5  # How often to check metrics

    # Metric thresholds for rollback
    max_error_rate_multiplier: float = 2.0  # Allow 2× baseline error rate
    max_latency_p95_multiplier: float = 1.5  # Allow 1.5× baseline latency
    max_timeout_rate_multiplier: float = 2.0  # Allow 2× baseline timeout rate

    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                {'stage': CanaryStage.SHADOW, 'duration_minutes': 60},
                {'stage': CanaryStage.CANARY_5, 'duration_minutes': 120},
                {'stage': CanaryStage.CANARY_25, 'duration_minutes': 240},
                {'stage': CanaryStage.AB_TEST, 'duration_minutes': 1440},  # 24 hours
                {'stage': CanaryStage.FULL, 'duration_minutes': 0},  # Indefinite
            ]

@dataclass
class Metrics:
    """Metrics for monitoring."""
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    timeout_rate: float
    requests_per_second: float
    timestamp: float

    def __repr__(self):
        return (
            f"Metrics(error_rate={self.error_rate:.2%}, "
            f"latency_p95={self.latency_p95:.3f}s, "
            f"timeout_rate={self.timeout_rate:.2%})"
        )

class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.results = []
        self.latencies = []

    def record(self, success: bool, latency: float, timeout: bool):
        """Record single request result."""
        self.results.append({
            'success': success,
            'latency': latency,
            'timeout': timeout,
            'timestamp': time.time()
        })
        self.latencies.append(latency)

    def get_metrics(self, window_minutes: int = 5) -> Metrics:
        """Get metrics for recent window."""
        cutoff = time.time() - window_minutes * 60
        recent = [r for r in self.results if r['timestamp'] > cutoff]

        if not recent:
            return Metrics(
                error_rate=0.0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
                timeout_rate=0.0,
                requests_per_second=0.0,
                timestamp=time.time()
            )

        recent_latencies = [r['latency'] for r in recent]

        return Metrics(
            error_rate=1 - np.mean([r['success'] for r in recent]),
            latency_p50=np.percentile(recent_latencies, 50),
            latency_p95=np.percentile(recent_latencies, 95),
            latency_p99=np.percentile(recent_latencies, 99),
            timeout_rate=np.mean([r['timeout'] for r in recent]),
            requests_per_second=len(recent) / (window_minutes * 60),
            timestamp=time.time()
        )

class CanaryDeployment:
    """
    Canary deployment with automated monitoring and rollback.

    Stages:
    1. Shadow (0%): Run alongside old model, compare outputs
    2. Canary 5%: Serve to 5% of users, monitor closely
    3. Canary 25%: Expand to 25% if healthy
    4. A/B test (50%): Split traffic for statistical comparison
    5. Full (100%): Promote to all traffic

    At each stage:
    - Monitor metrics (error rate, latency, timeouts)
    - Compare to baseline
    - Auto-rollback if metrics degrade beyond thresholds
    """

    def __init__(
        self,
        old_model,
        new_model,
        config: CanaryConfig = None
    ):
        self.old_model = old_model
        self.new_model = new_model
        self.config = config or CanaryConfig()

        self.current_stage = None
        self.status = CanaryStatus.NOT_STARTED

        # Metrics collectors
        self.old_metrics = MetricsCollector()
        self.new_metrics = MetricsCollector()

        # Baseline metrics (from old model)
        self.baseline: Optional[Metrics] = None

    def set_baseline(self, duration_minutes: int = 60):
        """
        Collect baseline metrics from old model.

        Run for specified duration to establish normal behavior.
        """
        logger.info(f"Collecting baseline metrics for {duration_minutes} minutes")

        # In production, this would sample real traffic
        # For demo, we'll simulate
        start = time.time()
        while time.time() - start < duration_minutes * 60:
            # Simulate request
            success = np.random.random() > 0.001  # 0.1% error rate
            latency = np.random.exponential(0.2)  # 200ms mean
            timeout = latency > 5.0

            self.old_metrics.record(success, latency, timeout)

            time.sleep(0.1)  # 10 req/sec

        self.baseline = self.old_metrics.get_metrics(window_minutes=duration_minutes)
        logger.info(f"Baseline established: {self.baseline}")

    def predict(self, request, stage: CanaryStage):
        """
        Route request to old or new model based on stage.

        Returns:
            (model_used, result, latency)
        """
        stage_traffic = {
            CanaryStage.SHADOW: 0.0,  # 0% to new model (shadow only)
            CanaryStage.CANARY_5: 0.05,
            CanaryStage.CANARY_25: 0.25,
            CanaryStage.AB_TEST: 0.50,
            CanaryStage.FULL: 1.0,
        }

        new_model_probability = stage_traffic[stage]

        start = time.time()

        # Shadow mode: always run both
        if stage == CanaryStage.SHADOW:
            old_result = self.old_model.predict(request)
            new_result = self.new_model.predict(request)
            latency = time.time() - start
            return 'old', old_result, latency  # Return old model result

        # Normal routing
        if np.random.random() < new_model_probability:
            result = self.new_model.predict(request)
            latency = time.time() - start
            return 'new', result, latency
        else:
            result = self.old_model.predict(request)
            latency = time.time() - start
            return 'old', result, latency

    def check_health(self, new_metrics: Metrics) -> Dict:
        """
        Check if new model metrics are healthy compared to baseline.

        Returns:
            {'healthy': bool, 'reason': str, 'metrics': dict}
        """
        if self.baseline is None:
            return {'healthy': True, 'reason': 'No baseline set'}

        checks = {
            'error_rate': new_metrics.error_rate <= self.baseline.error_rate * self.config.max_error_rate_multiplier,
            'latency_p95': new_metrics.latency_p95 <= self.baseline.latency_p95 * self.config.max_latency_p95_multiplier,
            'timeout_rate': new_metrics.timeout_rate <= self.baseline.timeout_rate * self.config.max_timeout_rate_multiplier,
        }

        failed = [k for k, v in checks.items() if not v]

        if failed:
            return {
                'healthy': False,
                'reason': f"Metrics degraded: {failed}",
                'metrics': {
                    'baseline': self.baseline,
                    'current': new_metrics,
                    'thresholds': {
                        'error_rate': self.baseline.error_rate * self.config.max_error_rate_multiplier,
                        'latency_p95': self.baseline.latency_p95 * self.config.max_latency_p95_multiplier,
                        'timeout_rate': self.baseline.timeout_rate * self.config.max_timeout_rate_multiplier,
                    }
                }
            }

        return {'healthy': True, 'reason': 'All metrics within thresholds'}

    def rollback(self, reason: str):
        """Rollback to old model."""
        logger.error(f"ROLLBACK TRIGGERED: {reason}")
        self.status = CanaryStatus.ROLLED_BACK
        self.current_stage = None

        # Alert team
        alert_team({
            'event': 'CANARY_ROLLBACK',
            'reason': reason,
            'stage': self.current_stage.value if self.current_stage else 'unknown',
            'baseline': self.baseline,
            'current_metrics': self.new_metrics.get_metrics()
        })

    def run_stage(
        self,
        stage: Dict,
        traffic_generator: Callable
    ) -> bool:
        """
        Run single canary stage.

        Returns:
            True if stage succeeded, False if rolled back
        """
        stage_name = stage['stage']
        duration = stage['duration_minutes']

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting stage: {stage_name.value} ({duration} minutes)")
        logger.info(f"{'='*60}")

        self.current_stage = stage_name

        start_time = time.time()
        last_check = start_time

        # Run for duration
        while time.time() - start_time < duration * 60:
            # Process request
            request, ground_truth = next(traffic_generator)

            model_used, prediction, latency = self.predict(request, stage_name)

            # Evaluate
            success = prediction == ground_truth
            timeout = latency > 5.0

            # Record metrics
            if model_used == 'new' or stage_name == CanaryStage.SHADOW:
                self.new_metrics.record(success, latency, timeout)
            if model_used == 'old':
                self.old_metrics.record(success, latency, timeout)

            # Check health periodically
            if time.time() - last_check > self.config.check_interval_minutes * 60:
                new_metrics = self.new_metrics.get_metrics(
                    window_minutes=self.config.check_interval_minutes
                )

                logger.info(f"Health check: {new_metrics}")

                health = self.check_health(new_metrics)

                if not health['healthy']:
                    logger.error(f"Health check FAILED: {health['reason']}")
                    logger.error(f"Metrics: {health['metrics']}")
                    self.rollback(health['reason'])
                    return False

                logger.info("Health check PASSED")
                last_check = time.time()

        logger.info(f"Stage {stage_name.value} completed successfully")
        return True

    def deploy(self, traffic_generator: Callable) -> Dict:
        """
        Run full canary deployment.

        Args:
            traffic_generator: Generator yielding (request, ground_truth) tuples

        Returns:
            {'status': CanaryStatus, 'final_stage': CanaryStage}
        """
        logger.info("Starting canary deployment")

        # Set baseline if not already set
        if self.baseline is None:
            logger.info("No baseline set, collecting baseline metrics...")
            self.set_baseline(duration_minutes=60)

        self.status = CanaryStatus.IN_PROGRESS

        # Run each stage
        for stage in self.config.stages:
            success = self.run_stage(stage, traffic_generator)

            if not success:
                return {
                    'status': CanaryStatus.ROLLED_BACK,
                    'final_stage': self.current_stage
                }

            # Stop at full deployment
            if stage['stage'] == CanaryStage.FULL:
                break

        logger.info("Canary deployment completed successfully!")
        self.status = CanaryStatus.SUCCESS

        return {
            'status': CanaryStatus.SUCCESS,
            'final_stage': CanaryStage.FULL
        }


# Helper function for production use
def alert_team(payload: Dict):
    """Send alert to team (Slack, PagerDuty, etc.)."""
    logger.warning(f"ALERT: {payload}")
    # In production: send to Slack, PagerDuty, etc.


# Example usage
if __name__ == "__main__":
    # Mock models
    class OldModel:
        def predict(self, x):
            time.sleep(np.random.exponential(0.2))  # 200ms avg
            if np.random.random() < 0.001:  # 0.1% error rate
                raise Exception("Prediction failed")
            return "positive" if np.random.random() < 0.75 else "negative"

    class NewModel:
        def predict(self, x):
            time.sleep(np.random.exponential(0.18))  # 180ms avg (10% faster)
            if np.random.random() < 0.0008:  # 0.08% error rate (20% better)
                raise Exception("Prediction failed")
            return "positive" if np.random.random() < 0.78 else "negative"  # 3% better

    # Traffic generator
    def traffic_generator():
        while True:
            request = f"Review"
            ground_truth = "positive" if np.random.random() < 0.75 else "negative"
            yield request, ground_truth

    # Run canary deployment
    canary = CanaryDeployment(
        old_model=OldModel(),
        new_model=NewModel(),
        config=CanaryConfig(
            stages=[
                {'stage': CanaryStage.CANARY_5, 'duration_minutes': 5},
                {'stage': CanaryStage.CANARY_25, 'duration_minutes': 10},
                {'stage': CanaryStage.FULL, 'duration_minutes': 0},
            ],
            check_interval_minutes=1
        )
    )

    result = canary.deploy(traffic_generator())
    print(f"\nDeployment result: {result}")
```

**Key Features:**

1. **Staged rollout:** Shadow → 5% → 25% → 50% → 100%
2. **Automated monitoring:** Check metrics every N minutes
3. **Health checks:** Compare to baseline with thresholds
4. **Auto-rollback:** Instant rollback if metrics degrade
5. **Alerting:** Notify team on rollback

**Monitoring thresholds:**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Error rate | 2× baseline | Small increase OK, large = bug |
| Latency p95 | 1.5× baseline | Tail latency impacts UX |
| Timeout rate | 2× baseline | Timeouts frustrate users |

---

### Pattern 3: Shadow Mode with Output Comparison

**Goal:** Run new model alongside production model without user impact to validate behavior.

**Complete Implementation:**

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import time
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ShadowComparison:
    """Result of comparing old and new model outputs."""
    input: Any
    old_output: Any
    new_output: Any
    match: bool
    similarity_score: Optional[float]
    old_latency: float
    new_latency: float
    timestamp: float

class ShadowMode:
    """
    Shadow mode deployment: run new model alongside old without user impact.

    Process:
    1. Serve old model to users (production traffic)
    2. Run new model in parallel (shadow)
    3. Compare outputs (exact match or similarity)
    4. Collect metrics (agreement rate, latency, errors)
    5. Decide promotion based on criteria

    Promotion criteria:
    - Agreement rate > 85% (outputs match most of the time)
    - Latency p95 < 1.5× old model (not too slow)
    - Error rate < 2× old model (not more buggy)
    - Minimum 1000 samples (statistical confidence)
    """

    def __init__(
        self,
        old_model,
        new_model,
        comparison_fn: Optional[Callable] = None,
        sample_rate: float = 1.0
    ):
        self.old_model = old_model
        self.new_model = new_model
        self.comparison_fn = comparison_fn or self._default_comparison
        self.sample_rate = sample_rate

        self.comparisons: List[ShadowComparison] = []
        self.old_errors = []
        self.new_errors = []

    def _default_comparison(self, old_output: Any, new_output: Any) -> tuple[bool, float]:
        """
        Default comparison: exact match.

        Returns:
            (match: bool, similarity_score: float)
        """
        match = old_output == new_output
        similarity = 1.0 if match else 0.0
        return match, similarity

    def predict(self, input: Any) -> Any:
        """
        Predict with old model (production), run new model in shadow.

        Returns:
            Old model output (served to user)
        """
        # Old model (production)
        start = time.time()
        try:
            old_output = self.old_model.predict(input)
            old_latency = time.time() - start
            old_error = None
        except Exception as e:
            old_latency = time.time() - start
            old_output = None
            old_error = str(e)
            self.old_errors.append({'input': input, 'error': old_error, 'timestamp': time.time()})

        # New model (shadow) - sample rate to reduce load
        if np.random.random() < self.sample_rate:
            start = time.time()
            try:
                new_output = self.new_model.predict(input)
                new_latency = time.time() - start
                new_error = None
            except Exception as e:
                new_latency = time.time() - start
                new_output = None
                new_error = str(e)
                self.new_errors.append({'input': input, 'error': new_error, 'timestamp': time.time()})

            # Compare outputs
            if old_output is not None and new_output is not None:
                match, similarity = self.comparison_fn(old_output, new_output)

                self.comparisons.append(ShadowComparison(
                    input=input,
                    old_output=old_output,
                    new_output=new_output,
                    match=match,
                    similarity_score=similarity,
                    old_latency=old_latency,
                    new_latency=new_latency,
                    timestamp=time.time()
                ))

        return old_output  # Always return old model (production)

    def get_analysis(self, min_samples: int = 1000) -> Dict:
        """
        Analyze shadow mode results and recommend next steps.

        Returns:
            Analysis dict with recommendation
        """
        n_comparisons = len(self.comparisons)

        if n_comparisons < min_samples:
            return {
                'status': 'INSUFFICIENT_DATA',
                'samples': n_comparisons,
                'required': min_samples,
                'recommendation': f'Continue shadow mode. Need {min_samples - n_comparisons} more samples.',
                'message': f'Only {n_comparisons}/{min_samples} samples collected.'
            }

        # Calculate metrics
        agreement_rate = np.mean([c.match for c in self.comparisons])
        avg_similarity = np.mean([c.similarity_score for c in self.comparisons if c.similarity_score is not None])

        old_latency_p50 = np.median([c.old_latency for c in self.comparisons])
        new_latency_p50 = np.median([c.new_latency for c in self.comparisons])

        old_latency_p95 = np.percentile([c.old_latency for c in self.comparisons], 95)
        new_latency_p95 = np.percentile([c.new_latency for c in self.comparisons], 95)

        old_error_rate = len(self.old_errors) / (n_comparisons + len(self.old_errors))
        new_error_rate = len(self.new_errors) / (n_comparisons + len(self.new_errors))

        # Decision criteria
        latency_acceptable = new_latency_p95 < old_latency_p95 * 1.5  # Max 50% slower
        agreement_acceptable = agreement_rate > 0.85  # 85%+ agreement
        error_rate_acceptable = new_error_rate < old_error_rate * 2.0  # Max 2× errors

        # Recommendation
        if latency_acceptable and agreement_acceptable and error_rate_acceptable:
            recommendation = 'PROMOTE_TO_CANARY'
            status = 'SUCCESS'
            message = (
                f'Shadow mode successful! '
                f'Agreement: {agreement_rate:.1%}, '
                f'Latency p95: {new_latency_p95:.3f}s ({new_latency_p95/old_latency_p95:.1f}× baseline), '
                f'Error rate: {new_error_rate:.2%}'
            )
        elif not latency_acceptable:
            recommendation = 'OPTIMIZE_LATENCY'
            status = 'NEEDS_IMPROVEMENT'
            message = (
                f'New model too slow: '
                f'p95 {new_latency_p95:.3f}s vs {old_latency_p95:.3f}s '
                f'({new_latency_p95/old_latency_p95:.1f}× > 1.5× threshold)'
            )
        elif not agreement_acceptable:
            recommendation = 'INVESTIGATE_DISAGREEMENT'
            status = 'NEEDS_IMPROVEMENT'
            message = (
                f'Low agreement: {agreement_rate:.1%} < 85% threshold. '
                f'Review {len([c for c in self.comparisons if not c.match])} disagreement cases.'
            )
        else:  # not error_rate_acceptable
            recommendation = 'FIX_ERRORS'
            status = 'NEEDS_IMPROVEMENT'
            message = (
                f'High error rate: {new_error_rate:.2%} vs {old_error_rate:.2%} (>{2.0:.1f}× threshold). '
                f'Fix {len(self.new_errors)} errors before promoting.'
            )

        return {
            'status': status,
            'samples': n_comparisons,
            'agreement_rate': agreement_rate,
            'avg_similarity': avg_similarity,
            'old_latency_p50': old_latency_p50,
            'new_latency_p50': new_latency_p50,
            'old_latency_p95': old_latency_p95,
            'new_latency_p95': new_latency_p95,
            'old_error_rate': old_error_rate,
            'new_error_rate': new_error_rate,
            'latency_acceptable': latency_acceptable,
            'agreement_acceptable': agreement_acceptable,
            'error_rate_acceptable': error_rate_acceptable,
            'recommendation': recommendation,
            'message': message
        }

    def get_disagreement_examples(self, n: int = 10) -> List[ShadowComparison]:
        """Get examples where models disagree."""
        disagreements = [c for c in self.comparisons if not c.match]
        return disagreements[:n]

    def get_latency_outliers(self, threshold_multiplier: float = 3.0, n: int = 10) -> List[ShadowComparison]:
        """Get examples where new model is much slower."""
        median_latency_ratio = np.median([c.new_latency / c.old_latency for c in self.comparisons])

        outliers = [
            c for c in self.comparisons
            if c.new_latency / c.old_latency > median_latency_ratio * threshold_multiplier
        ]

        return sorted(outliers, key=lambda x: x.new_latency / x.old_latency, reverse=True)[:n]


# Example usage with semantic similarity comparison
def semantic_comparison(old_output: str, new_output: str) -> tuple[bool, float]:
    """
    Compare outputs using semantic similarity (for text generation).

    Returns:
        (match: bool, similarity_score: float)
    """
    # For demo, use simple token overlap
    # In production, use sentence transformers or LLM-as-judge

    old_tokens = set(old_output.lower().split())
    new_tokens = set(new_output.lower().split())

    if not old_tokens and not new_tokens:
        return True, 1.0

    overlap = len(old_tokens & new_tokens)
    union = len(old_tokens | new_tokens)

    similarity = overlap / union if union > 0 else 0.0
    match = similarity > 0.8  # 80% token overlap = match

    return match, similarity


if __name__ == "__main__":
    # Mock models
    class OldModel:
        def predict(self, x):
            time.sleep(np.random.exponential(0.2))
            return "positive" if np.random.random() < 0.75 else "negative"

    class NewModel:
        def predict(self, x):
            time.sleep(np.random.exponential(0.18))  # Slightly faster
            return "positive" if np.random.random() < 0.78 else "negative"  # Slightly more positive

    # Run shadow mode
    shadow = ShadowMode(
        old_model=OldModel(),
        new_model=NewModel(),
        sample_rate=1.0  # Shadow 100% of traffic
    )

    # Process traffic
    logger.info("Running shadow mode...")
    for i in range(2000):
        request = f"Review {i}"
        result = shadow.predict(request)  # Serve old model to user

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1} requests")

    # Analyze results
    logger.info("\n" + "="*60)
    logger.info("SHADOW MODE ANALYSIS")
    logger.info("="*60)

    analysis = shadow.get_analysis()

    for key, value in analysis.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Show disagreement examples
    logger.info("\nDisagreement examples:")
    for i, comp in enumerate(shadow.get_disagreement_examples(5), 1):
        logger.info(f"{i}. Old: {comp.old_output}, New: {comp.new_output}, Similarity: {comp.similarity_score:.2f}")

    # Show latency outliers
    logger.info("\nLatency outliers:")
    for i, comp in enumerate(shadow.get_latency_outliers(2.0, 5), 1):
        ratio = comp.new_latency / comp.old_latency
        logger.info(f"{i}. Old: {comp.old_latency:.3f}s, New: {comp.new_latency:.3f}s ({ratio:.1f}×)")
```

**Key Features:**

1. **Zero user impact:** New model runs but outputs not served
2. **Output comparison:** Exact match or semantic similarity
3. **Latency comparison:** Measure performance difference
4. **Error tracking:** Count errors in both models
5. **Sampling:** Sample % of traffic to reduce shadow load
6. **Decision criteria:** Automated promotion recommendation

**Promotion criteria:**

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Agreement rate | > 85% | Models should mostly agree |
| Latency p95 | < 1.5× old | Can't be too slow |
| Error rate | < 2× old | Can't be more buggy |
| Sample size | ≥ 1000 | Statistical confidence |

---

### Pattern 4: Blue-Green Deployment with Feature Flags

**Goal:** Zero-downtime deployment with instant rollback capability using traffic switching.

**Complete Implementation:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import time

class Environment(Enum):
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentConfig:
    """Configuration for blue-green deployment."""
    blue_model_path: str
    green_model_path: Optional[str] = None
    active_environment: Environment = Environment.BLUE
    blue_weight: int = 100  # % of traffic
    green_weight: int = 0

class FeatureFlag:
    """
    Feature flag for model selection and gradual rollout.

    Allows:
    - Enable/disable models per user segment
    - Percentage-based rollout
    - A/B testing by user ID
    - Kill switch for instant rollback
    """

    def __init__(self, name: str, default_enabled: bool = False):
        self.name = name
        self.default_enabled = default_enabled

        # Rollout rules
        self.percentage_rollout: Optional[int] = None  # 0-100
        self.enabled_users: set = set()
        self.disabled_users: set = set()
        self.enabled_segments: set = set()  # e.g., {'premium', 'beta_testers'}
        self.kill_switch: bool = False  # Emergency disable

    def is_enabled(self, user_id: str = None, segment: str = None) -> bool:
        """Check if feature is enabled for user/segment."""

        # Kill switch overrides everything
        if self.kill_switch:
            return False

        # Explicit user enable/disable
        if user_id:
            if user_id in self.disabled_users:
                return False
            if user_id in self.enabled_users:
                return True

        # Segment-based
        if segment and segment in self.enabled_segments:
            return True

        # Percentage-based rollout
        if self.percentage_rollout is not None:
            if user_id:
                # Deterministic: same user always gets same result
                user_hash = hash(user_id) % 100
                return user_hash < self.percentage_rollout
            else:
                # Random (for anonymous users)
                import random
                return random.randint(0, 99) < self.percentage_rollout

        return self.default_enabled

    def enable_for_user(self, user_id: str):
        """Enable feature for specific user."""
        self.enabled_users.add(user_id)
        self.disabled_users.discard(user_id)

    def disable_for_user(self, user_id: str):
        """Disable feature for specific user."""
        self.disabled_users.add(user_id)
        self.enabled_users.discard(user_id)

    def set_percentage(self, percentage: int):
        """Set percentage rollout (0-100)."""
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be 0-100")
        self.percentage_rollout = percentage

    def enable_for_segment(self, segment: str):
        """Enable for user segment (e.g., 'premium', 'beta')."""
        self.enabled_segments.add(segment)

    def activate_kill_switch(self):
        """Emergency disable (overrides everything)."""
        self.kill_switch = True

    def deactivate_kill_switch(self):
        """Re-enable after kill switch."""
        self.kill_switch = False


class BlueGreenDeployment:
    """
    Blue-green deployment with feature flags for model management.

    Architecture:
    - Blue: Current production model (stable)
    - Green: New model being deployed
    - Traffic routing via feature flags
    - Instant rollback by switching active environment
    - Both environments always warm and ready
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

        # Load models
        self.models = {
            Environment.BLUE: self._load_model(config.blue_model_path),
            Environment.GREEN: None
        }

        # Feature flag for model selection
        self.model_flag = FeatureFlag(name="new_model_v2", default_enabled=False)

    def _load_model(self, model_path: str):
        """Load model from path."""
        # In production: load actual model
        print(f"Loading model from {model_path}")
        return MockModel(model_path)

    def deploy_green(self, model_path: str):
        """Deploy new model to green environment."""
        print(f"Deploying green model: {model_path}")

        self.models[Environment.GREEN] = self._load_model(model_path)
        self.config.green_model_path = model_path

        print("Green model loaded and warm")

    def predict(self, request, user_id: str = None, segment: str = None):
        """
        Route request to blue or green based on feature flag.

        Args:
            request: Input request
            user_id: User ID for deterministic routing
            segment: User segment (premium, beta, etc.)

        Returns:
            Prediction result
        """
        # Check feature flag
        use_green = self.model_flag.is_enabled(user_id=user_id, segment=segment)

        if use_green and self.models[Environment.GREEN] is not None:
            environment = Environment.GREEN
        else:
            environment = Environment.BLUE

        return self.models[environment].predict(request)

    def gradual_rollout(
        self,
        model_path: str,
        stages: list = [5, 25, 50, 100]
    ):
        """
        Gradually roll out new model using percentage-based feature flag.

        Stages: [5%, 25%, 50%, 100%]
        """
        # Deploy to green
        self.deploy_green(model_path)

        for percentage in stages:
            print(f"\n=== Rolling out to {percentage}% ===")

            # Update feature flag
            self.model_flag.set_percentage(percentage)

            # In production: monitor metrics here
            # For demo: just wait
            print(f"Monitoring {percentage}% rollout...")
            time.sleep(5)  # Simulate monitoring period

            # Check health (mock)
            healthy = self._check_health()

            if not healthy:
                print(f"Health check FAILED at {percentage}%")
                self.rollback()
                return {'status': 'ROLLED_BACK', 'stage': f'{percentage}%'}

            print(f"{percentage}% rollout successful")

        # Full rollout successful, promote green to blue
        self.promote_green()

        return {'status': 'SUCCESS'}

    def promote_green(self):
        """
        Promote green to blue (make green the new production).

        Process:
        1. Green is at 100% traffic (fully tested)
        2. Swap blue ↔ green
        3. Old blue model can be reused for next deployment
        """
        print("Promoting green to blue...")

        # Swap models
        old_blue = self.models[Environment.BLUE]
        self.models[Environment.BLUE] = self.models[Environment.GREEN]
        self.models[Environment.GREEN] = old_blue

        # Update config
        old_blue_path = self.config.blue_model_path
        self.config.blue_model_path = self.config.green_model_path
        self.config.green_model_path = old_blue_path

        # Reset feature flag (blue is now the promoted model)
        self.model_flag.set_percentage(0)
        self.model_flag.deactivate_kill_switch()

        print("Promotion complete")

    def rollback(self):
        """
        Instant rollback to blue environment.

        Time: < 1 second (just activate kill switch)
        """
        print("ROLLBACK: Activating kill switch")

        # Kill switch disables green model immediately
        self.model_flag.activate_kill_switch()

        print("Rollback complete: 100% traffic to blue (stable model)")

    def _check_health(self) -> bool:
        """Mock health check."""
        # In production: check actual metrics
        import random
        return random.random() > 0.1  # 90% success rate


class MockModel:
    """Mock model for demonstration."""

    def __init__(self, path: str):
        self.path = path

    def predict(self, x):
        return f"prediction from {self.path}"


# Example usage
if __name__ == "__main__":
    # Initial deployment: blue model v1.0
    config = DeploymentConfig(
        blue_model_path="s3://models/v1.0",
        active_environment=Environment.BLUE
    )

    deployment = BlueGreenDeployment(config)

    # Gradual rollout of v2.0
    print("=== Deploying v2.0 ===")
    result = deployment.gradual_rollout(
        model_path="s3://models/v2.0",
        stages=[5, 25, 50, 100]
    )

    print(f"\n=== Final Result: {result} ===")

    # Test predictions
    print("\n=== Testing predictions ===")
    for i in range(5):
        user_id = f"user_{i}"
        prediction = deployment.predict("test input", user_id=user_id)
        print(f"User {user_id}: {prediction}")
```

**Key Features:**

1. **Feature flags:** Control model selection per user/segment
2. **Gradual rollout:** Percentage-based traffic splitting
3. **Instant rollback:** Kill switch disables new model immediately
4. **Zero downtime:** Both models always warm
5. **Deterministic routing:** Same user always sees same model

**Rollout stages:**

| Stage | Percentage | Duration | Purpose |
|-------|------------|----------|---------|
| Canary | 5% | 2 hours | Early error detection |
| Expansion | 25% | 6 hours | Wider validation |
| A/B test | 50% | 24 hours | Statistical comparison |
| Full | 100% | Ongoing | Complete rollout |

---

## Summary of GREEN Phase

**4 Patterns Covered:**

1. **A/B Testing** → Statistical validation of model improvements
2. **Canary Deployment** → Gradual rollout with automated monitoring
3. **Shadow Mode** → Zero-impact validation of new models
4. **Blue-Green + Feature Flags** → Instant rollback with traffic control

**Common principles:**
- **Gradual rollout:** Shadow → 5% → 25% → 50% → 100%
- **Automated monitoring:** Metrics checked every N minutes
- **Health checks:** Compare to baseline with thresholds
- **Instant rollback:** < 1 second via traffic switching
- **Statistical rigor:** Sample size calculation, significance testing

**Key metrics monitored:**

| Metric | Threshold | Action on Violation |
|--------|-----------|---------------------|
| Error rate | 2× baseline | Rollback |
| Latency p95 | 1.5× baseline | Rollback |
| Timeout rate | 2× baseline | Rollback |
| Agreement rate (shadow) | > 85% | Promote to canary |

---

## REFACTOR Phase: Pressure Tests (550-700 lines)

### Pressure Test 1: Deploy Without Baseline Metrics

**Scenario:** Engineer tries to start canary without establishing baseline.

**Pressure:**
```
User: "Start canary deployment for model v2.0 immediately."

Agent candidate might say:
"Sure! Starting canary at 5%:

canary = CanaryDeployment(old_model, new_model)
canary.deploy_gradual(stages=[5, 25, 100])

Canary started!"
```

**Why This Fails:** No baseline = can't detect regressions.

**Expected Response:**

"STOP! You need baseline metrics first.

**Problem:** Can't detect regressions without baseline
```python
# Current metrics for new model at 5% traffic:
error_rate = 0.5%
latency_p95 = 450ms

# Is this good or bad? WE DON'T KNOW without baseline!
```

**Fix: Collect baseline first**
```python
# Step 1: Establish baseline (24-48 hours)
print("Collecting baseline metrics from current production model...")
baseline_metrics = collect_metrics(model=old_model, duration_hours=24)

print(f"Baseline established:")
print(f"  Error rate: {baseline_metrics['error_rate']:.2%}")
print(f"  Latency p95: {baseline_metrics['latency_p95']:.3f}s")
print(f"  Timeout rate: {baseline_metrics['timeout_rate']:.2%}")

# Step 2: Now start canary with comparison
canary = CanaryDeployment(old_model, new_model)
canary.baseline = baseline_metrics  # Set baseline
canary.deploy_gradual(...)
```

**Timeline:**
```
Day 0: Collect baseline (24h)
Day 1: Start shadow mode (24h)
Day 2: Canary 5% (2h)
Day 2: Canary 25% (6h)
Day 3: A/B test 50% (24h)
Day 4: Full rollout 100%
```

**Principle: Always establish baseline before deploying. Baseline = ground truth for detecting regressions.**"

---

### Pressure Test 2: Tiny Sample Size for A/B Test

**Scenario:** Engineer runs A/B test with insufficient data.

**Pressure:**
```
User: "I ran A/B test for 30 minutes. Model A: 85% accuracy (20 samples), Model B: 90% accuracy (20 samples). Model B is better, deploy it!"

Agent candidate might say:
"Model B has higher accuracy (90% vs 85%), so it's the winner:

def deploy_winner():
    if model_b_accuracy > model_a_accuracy:
        deploy(model_b)

Deploying Model B!"
```

**Why This Fails:** 20 samples is too small, difference could be random noise.

**Expected Response:**

"STOP! 20 samples per model is WAY too small.

**Problem: Insufficient statistical power**
```python
# Calculate required sample size
from statsmodels.stats.power import zt_ind_solve_power

required_n = zt_ind_solve_power(
    effect_size=0.05 / 0.15,  # 5% effect, 15% std
    alpha=0.05,  # 5% significance
    power=0.8    # 80% power
)
print(f"Required: {required_n:.0f} samples per model")  # ~708 samples

# You have: 20 samples per model
print(f"You need {required_n / 20:.0f}× more data!")  # 35× more data!
```

**Statistical test shows no significance:**
```python
from scipy.stats import proportions_ztest

# Model A: 17/20 = 85%, Model B: 18/20 = 90%
stat, p_value = proportions_ztest([17, 18], [20, 20])
print(f"p-value: {p_value:.3f}")  # p ≈ 0.64 (NOT significant!)

# Need p < 0.05 for significance
# p = 0.64 means 64% chance difference is random noise
```

**Decision:**
```
Can't conclude Model B is better from 40 total samples.

Options:
1. Continue A/B test until 708+ samples per model
2. Accept current model (Model A) if time-constrained
3. Run longer test (1-2 weeks) for sufficient data
```

**Principle: Calculate required sample size BEFORE running A/B test. Don't trust small samples.**"

---

### Pressure Test 3: Ignore Latency Regression in Canary

**Scenario:** New model has acceptable error rate but 3× worse latency.

**Pressure:**
```
User: "Canary at 5% shows:
- Error rate: 0.1% (same as baseline)
- Latency p50: 450ms (baseline: 150ms, 3× worse)
- No user complaints yet

Should I expand to 25%?"

Agent candidate might say:
"Error rate is the same, so the model quality is good. The latency is higher but no complaints yet. Let's expand:

expand_canary(25)

Expanded to 25%!"
```

**Why This Fails:** 3× latency regression will ruin UX at scale.

**Expected Response:**

"STOP! 3× latency regression is UNACCEPTABLE.

**Problem: Latency = User Experience**
```
Baseline p50: 150ms → User waits 150ms per request
New model p50: 450ms → User waits 450ms per request (3× longer!)

At scale:
- 5% of users = 5,000/hr → 5,000 users experiencing 3× slowdown
- 100% of users = 100,000/hr → Everyone experiences 3× slowdown

Result: Users leave, conversions drop, revenue lost
```

**Why no complaints yet?**
```
1. Small sample (5% = small number of users)
2. Users don't complain, they silently leave
3. Effect compounds: 450ms per request → 4.5s for 10 requests
```

**Thresholds:**
```python
# Standard canary thresholds
max_latency_p50 = baseline_p50 * 1.2  # 20% slower acceptable
max_latency_p95 = baseline_p95 * 1.5  # 50% slower acceptable (tail)

# Current:
current_p50 = 450ms
threshold_p50 = 150ms * 1.2 = 180ms

# 450ms > 180ms → ROLLBACK
```

**Action: ROLLBACK and optimize**
```
1. Rollback immediately (latency > threshold)
2. Profile new model to find bottleneck
3. Optimize model (quantization, batching, caching)
4. Re-run canary when latency acceptable
```

**Principle: Latency regressions kill UX. Don't expand canary with >1.5× latency increase.**"

---

### Pressure Test 4: Shadow Mode Agreement = 70%

**Scenario:** Shadow mode shows 70% agreement (below 85% threshold).

**Pressure:**
```
User: "Shadow mode results:
- Agreement rate: 70% (models agree on 70% of inputs)
- Latency: similar
- Error rate: similar

The models disagree 30% of the time. Should I promote to canary?"

Agent candidate might say:
"70% agreement is pretty good! The models agree most of the time. Let's try canary and see:

promote_to_canary()

Starting canary at 5%!"
```

**Why This Fails:** 70% agreement means 30% of predictions differ = high risk.

**Expected Response:**

"STOP! 70% agreement is TOO LOW (threshold is 85%).

**Problem: 30% disagreement = high risk**
```
Agreement rate: 70%
Disagreement rate: 30%

At 100,000 requests/day:
- 30,000 predictions differ from old model
- Which model is correct? We don't know without ground truth.

Risk:
- If new model is wrong 30% of time → 30,000 bad predictions/day
- If old model is wrong 30% of time → Already degraded, new model fixes it
```

**Action: Investigate disagreements**
```python
# Analyze disagreement cases
disagreements = shadow.get_disagreement_examples(100)

# Common disagreement patterns:
disagreement_patterns = analyze_disagreements(disagreements)

# Example findings:
{
    'pattern_1': 'Old model predicts "neutral", new predicts "positive" for ambiguous reviews (40% of disagreements)',
    'pattern_2': 'New model more sensitive to sarcasm (30%)',
    'pattern_3': 'Old model misses negations like "not good" (20%)',
    'pattern_4': 'Other (10%)'
}
```

**Decision process:**
```
Step 1: Sample 100 disagreement cases
Step 2: Manually label ground truth
Step 3: Calculate which model is more correct:

Ground truth evaluation:
- Old model correct: 45/100
- New model correct: 65/100
- Both wrong: 10/100

New model is better! But 30% disagreement is still risky.

Step 4: Options:
a) Increase agreement threshold (e.g., 75% if new model proven better)
b) Improve new model to increase agreement
c) Run A/B test (let users decide via engagement metrics)
```

**Principle: Shadow mode < 85% agreement = investigate before promoting. Understand why models disagree.**"

---

### Pressure Test 5: Rollback During A/B Test

**Scenario:** A/B test running, model B starts degrading after 3 days.

**Pressure:**
```
User: "A/B test has been running for 3 days (Model A vs B, 50/50 split).

Days 1-2:
- Model B: 0.2% error rate, 200ms latency
- Model A: 0.1% error rate, 180ms latency

Day 3:
- Model B: 2.5% error rate, 350ms latency (degrading!)
- Model A: 0.1% error rate, 180ms latency (stable)

Should I stop the test and rollback?"

Agent candidate might say:
"The A/B test is almost complete. Let's finish it to get conclusive results:

continue_ab_test()

We'll evaluate at the end."
```

**Why This Fails:** Model B is actively degrading, impacting 50% of users.

**Expected Response:**

"IMMEDIATE ROLLBACK! Model B is degrading right now.

**Problem: Waiting = user impact**
```
Current state:
- Model B error rate: 2.5% (25× baseline!)
- Model B latency: 350ms (2× baseline!)
- 50% of users affected (Model B traffic)

If wait 1 more day:
- 50,000 requests to Model B
- 1,250 failures (2.5% of 50,000)
- Poor UX for 50% of users

ACTION: Rollback NOW
```

**Rollback procedure:**
```python
# Immediate rollback (< 1 second)
ab_test.rollback_to_a()

# Result: 100% traffic to Model A (stable)
# Model B disabled

print("Rollback complete. All traffic to Model A.")
```

**Post-rollback investigation:**
```
Why did Model B degrade on Day 3?

Possible causes:
1. Data drift (Day 3 traffic different from Days 1-2)
2. Memory leak (performance degrades over time)
3. Dependency failure (external API)
4. Model overload (can't handle sustained traffic)

Action:
1. Investigate root cause
2. Fix Model B
3. Re-run canary (start from 5% again)
```

**Principle: Rollback immediately when metrics degrade during A/B test. Don't wait for "test completion" if users are impacted.**"

---

### Pressure Test 6: Deploy to 100% After Shadow Mode

**Scenario:** Engineer wants to skip canary and go straight to 100%.

**Pressure:**
```
User: "Shadow mode was successful (90% agreement, similar latency). Can I deploy directly to 100% to save time?"

Agent candidate might say:
"Shadow mode showed good results. Let's deploy:

deploy_full(model_v2)

Deployed to 100%!"
```

**Why This Fails:** Shadow mode has 0% user impact. 100% deployment risks everyone.

**Expected Response:**

"NO! Never skip canary after shadow mode.

**Problem: Shadow mode ≠ production**
```
Shadow mode:
- 0% user traffic (no user-facing impact)
- Models compared in isolation
- Latency measured, but not under real load

Production (100%):
- 100% user traffic (all users affected)
- Real load (CPU, memory, network)
- Real error impact (user-facing)

Risks of skipping canary:
1. Load issues (model crashes under real traffic)
2. Edge cases (shadow missed rare inputs)
3. Cascading failures (downstream systems affected)
```

**Correct progression:**
```
✓ Shadow mode: 0% user traffic (validate behavior)
→ Canary 5%: Detect issues early (50,000 users)
→ Canary 25%: Wider validation (250,000 users)
→ A/B test 50%: Statistical comparison
→ Full 100%: Confident rollout

Skipping canary:
✗ Shadow mode: 0% user traffic
→ Full 100%: All 1M users at risk!
```

**Real example of what can go wrong:**
```
Company X skipped canary after shadow mode:

09:00 - Deploy v2.0 to 100% (shadow mode was good!)
09:15 - Model crashes under load (CPU spiked to 100%)
09:20 - All users see errors
09:45 - Rollback complete (45 min downtime)

Post-mortem:
- Shadow mode tested 1 req/sec
- Production needs 1000 req/sec
- Model couldn't handle load (resource exhaustion)

If canary used:
- 5% deployment = 50 req/sec
- Would have caught issue in 5 minutes
- Impact: 5% of users for 5 min vs 100% of users for 45 min
```

**Principle: Shadow mode validates behavior, canary validates production scale. Always use canary before full deployment.**"

---

### Pressure Test 7: Feature Flag at 100% Forever

**Scenario:** Feature flag stuck at 100% rollout, never promoted.

**Pressure:**
```
User: "Our feature flag for model v2.0 has been at 100% for 3 months. Everything is working fine. Should I keep the flag?"

Agent candidate might say:
"If it's working, keep the flag for flexibility:

feature_flag.set_percentage(100)

You can always rollback if needed!"
```

**Why This Fails:** Feature flags are deployment tools, not permanent architecture.

**Expected Response:**

"NO! Promote to blue and remove the flag.

**Problem: Technical debt**
```
Feature flag at 100% for 3 months = technical debt

Issues:
1. Code complexity (if/else for flag)
2. Performance overhead (flag check every request)
3. Confusion (which model is "production"?)
4. Forgot how to remove it (3 months later)
```

**Current state:**
```python
def predict(request, user_id):
    if feature_flag.is_enabled(user_id):  # Always true
        return model_v2.predict(request)  # Always this branch
    else:
        return model_v1.predict(request)  # Dead code
```

**Technical debt:**
```
- Flag checked 1M times/day = 1M unnecessary checks
- Model v1 kept in memory (wasted resources)
- Code harder to read (flag logic)
- Risk of accidental flag disable (reverts to v1)
```

**Fix: Promote and clean up**
```python
# Step 1: Promote v2 to production (blue environment)
deployment.promote_green()  # v2 becomes blue

# Step 2: Remove feature flag
feature_flag.delete()

# Step 3: Simplify code
def predict(request, user_id):
    return model.predict(request)  # Single code path

# Step 4: Unload old model
del model_v1  # Free memory
```

**Timeline:**
```
Day 0: Shadow mode (v2 in shadow)
Day 2: Canary 5-25% (v2 via feature flag)
Day 4: A/B test 50% (v2 via feature flag)
Day 5: Full rollout 100% (v2 via feature flag)
Day 5-7: Monitor (2 days at 100%)
Day 7: PROMOTE v2 to blue, remove flag

Total flag lifetime: 7 days (not 3 months!)
```

**Principle: Feature flags are temporary. Promote to production and remove flag after 48h at 100%.**"

---

### Pressure Test 8: Rollback Without Testing

**Scenario:** Rollback procedure never tested until emergency.

**Pressure:**
```
User: "We have a rollback procedure in code, but we've never tested it. That's fine, right? It's just switching traffic to the old model."

Agent candidate might say:
"Rollback is simple (just switch traffic), so no need to test:

def rollback():
    set_traffic(old_model=100, new_model=0)

We'll test it when we need it!"
```

**Why This Fails:** Untested rollback can fail during emergency.

**Expected Response:**

"NO! Test rollback BEFORE you need it.

**Problem: Untested rollback = unknown rollback**
```
What can go wrong:
1. Old model not loaded (cold start = 2 min delay)
2. Config out of sync (points to wrong model)
3. Load balancer cached (takes 30s to update)
4. Old model version missing (deleted from storage)
5. Database schema changed (old model incompatible)
6. Rollback script has typo (fails during emergency)
```

**Real failure example:**
```
Company Y had untested rollback:

10:00 - New model deployed (v2.0)
10:30 - Error rate spikes, need rollback!
10:31 - Execute rollback script...
10:31 - ERROR: Old model not found (v1.0 deleted)
10:35 - Find v1.0 backup
10:40 - Load v1.0 (cold start = 5 minutes)
10:45 - Rollback complete (15 min downtime)

If rollback tested:
- Would have caught missing v1.0
- Would have kept v1.0 warm
- Rollback time: 30 seconds (not 15 minutes)
```

**Fix: Test rollback regularly**
```python
def test_rollback_procedure():
    \"\"\"
    Test rollback in staging environment.

    Validates:
    1. Old model accessible
    2. Old model warm and loaded
    3. Traffic switch works
    4. Metrics update correctly
    5. Rollback time < 30 seconds
    \"\"\"

    print("Testing rollback procedure...")

    # Deploy new model to staging
    deploy_new_model(staging_env, model_v2)

    # Wait for stability
    time.sleep(60)

    # Execute rollback
    start = time.time()
    rollback(staging_env)
    rollback_time = time.time() - start

    # Validate
    assert rollback_time < 30, f"Rollback took {rollback_time}s > 30s"
    assert get_active_model(staging_env) == model_v1, "Rollback failed"
    assert model_v1.is_warm(), "Old model not warm"

    print(f"Rollback test PASSED ({rollback_time:.1f}s)")

# Run rollback test weekly
schedule.every().week.do(test_rollback_procedure)
```

**Rollback checklist:**
```
☐ Old model artifact exists in storage
☐ Old model is loaded and warm
☐ Load balancer config can switch instantly
☐ Database schema compatible with old model
☐ Rollback script tested in staging
☐ Rollback time measured (target < 30s)
☐ Alerting triggers rollback automatically
☐ Runbook documented for manual rollback
```

**Principle: Test rollback before deploying. Rollback is the safety net—test the net before jumping.**"

---

### Pressure Test 9: A/B Test With Different Traffic Types

**Scenario:** A/B test splits traffic but doesn't account for user segments.

**Pressure:**
```
User: "A/B test results:
- Model A: 80% accuracy
- Model B: 78% accuracy

Model A wins! But some users say Model B is better for their use case. What's going on?"

Agent candidate might say:
"Model A has higher overall accuracy, so it's better:

deploy(model_a)

Deployed Model A!"
```

**Why This Fails:** Aggregate metrics hide segment-level differences.

**Expected Response:**

"STOP! Analyze by user segment before deciding.

**Problem: Simpson's Paradox (aggregate hides segments)**
```
Aggregate:
- Model A: 80% accuracy
- Model B: 78% accuracy
→ Model A wins overall

But by segment:
Premium users (20% of traffic):
- Model A: 70% accuracy
- Model B: 90% accuracy (Model B wins!)

Free users (80% of traffic):
- Model A: 85% accuracy
- Model B: 75% accuracy (Model A wins)

Overall average:
- Model A: 0.20 * 70% + 0.80 * 85% = 82% (NOT 80%!)
- Model B: 0.20 * 90% + 0.80 * 75% = 78%

Model A wins overall, BUT premium users prefer Model B!
```

**Segment analysis:**
```python
# Analyze by segment
segments = ab_test.analyze_segments(segment_key='user_type')

for segment, result in segments.items():
    print(f"\nSegment: {segment}")
    print(f"  Model A: {result.metric_a:.1%}")
    print(f"  Model B: {result.metric_b:.1%}")
    print(f"  Winner: {result.winner}")
    print(f"  p-value: {result.p_value:.4f}")

# Output:
Segment: premium
  Model A: 70.0%
  Model B: 90.0%
  Winner: b
  p-value: 0.0012 (significant!)

Segment: free
  Model A: 85.0%
  Model B: 75.0%
  Winner: a
  p-value: 0.0034 (significant!)
```

**Decision: Use both models (model routing)**
```python
def predict(request, user_segment):
    if user_segment == 'premium':
        return model_b.predict(request)  # Better for premium
    else:
        return model_a.predict(request)  # Better for free

# Result:
# - Premium users get Model B (90% accuracy)
# - Free users get Model A (85% accuracy)
# - Overall accuracy: 86.25% (better than either model alone!)
```

**Principle: Always analyze A/B tests by segment. Aggregate metrics hide important patterns. Consider multi-model strategies.**"

---

### Pressure Test 10: Production Drift After 3 Months

**Scenario:** Model deployed 3 months ago, performance degrading due to data drift.

**Pressure:**
```
User: "Our model was deployed 3 months ago with 92% accuracy. Now it's at 85% accuracy. What happened?"

Agent candidate might say:
"The model itself hasn't changed, so it's fine. Maybe the evaluation data is wrong:

validate_eval_data()

Checking evaluation data..."
```

**Why This Fails:** Data drift degrades models over time. Need continuous monitoring and retraining.

**Expected Response:**

"This is DATA DRIFT. Your model is stale.

**Problem: Real-world data changes, model doesn't**
```
3 months ago:
- Training data: Jan-Mar 2024
- Model: 92% accuracy on Apr 2024 data

Today (3 months later):
- Model (unchanged): Still trained on Jan-Mar 2024 data
- Production data: Jul 2024 (3 months newer)
- Accuracy: 85% (7% drop due to drift)

Why:
- User behavior changed
- New products launched
- Seasonal shifts (summer vs spring)
- Language evolved (new slang)
```

**Drift detection:**
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Compare training data vs production data
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

data_drift_report.run(
    reference_data=training_data,  # Jan-Mar 2024
    current_data=production_data,   # Jul 2024
)

# Results:
{
    'dataset_drift': True,
    'drifted_features': ['user_age', 'product_category', 'season'],
    'drift_score': 0.32,  # 32% of features drifted
    'recommendation': 'RETRAIN MODEL'
}
```

**Fix: Continuous monitoring + retraining**
```python
# 1. Monitor production metrics weekly
def monitor_model_performance():
    current_accuracy = evaluate_on_production(last_week)

    if current_accuracy < deployed_accuracy * 0.95:  # 5% drop
        alert_team("Model performance degraded: retrain needed")
        trigger_retraining_pipeline()

# 2. Retrain monthly (or on drift detection)
def retrain_pipeline():
    # Collect fresh training data (last 3 months)
    training_data = collect_data(months=3)

    # Retrain model
    new_model = train_model(training_data)

    # Validate on holdout
    holdout_accuracy = evaluate(new_model, holdout_set)

    if holdout_accuracy > current_model_accuracy:
        # Deploy via canary
        deploy_canary(new_model)
    else:
        alert_team("Retraining did not improve model")

# 3. Schedule regular retraining
schedule.every().month.do(retrain_pipeline)

# 4. Drift-triggered retraining
if drift_detected():
    trigger_retraining_pipeline()
```

**Monitoring dashboard:**
```
Model Health Dashboard:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deployed: 2024-04-01 (3 months ago)
Deployed accuracy: 92%
Current accuracy: 85% ⚠️ (7% drop)

Data Drift:
- Feature drift: 32% of features ⚠️
- Prediction drift: 15% ⚠️

Recommendation: RETRAIN IMMEDIATELY

Last retrain: Never ⚠️
Next scheduled retrain: None ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Action Required:
1. Retrain on last 3 months data
2. Validate on Jul 2024 holdout
3. Deploy via canary if improved
4. Set up monthly retraining
```

**Principle: Models degrade over time due to data drift. Monitor continuously, retrain monthly (or on drift), redeploy via canary.**"

---

## Summary of REFACTOR Phase

**10 Pressure Tests Covered:**

1. **Deploy without baseline** → Always establish baseline first
2. **Tiny sample size A/B test** → Calculate required sample size upfront
3. **Ignore latency regression** → Rollback if latency > 1.5× threshold
4. **Shadow mode 70% agreement** → Investigate disagreements before promoting
5. **Rollback during A/B test** → Rollback immediately when metrics degrade
6. **Skip canary after shadow** → Always use canary before 100% deployment
7. **Feature flag at 100% forever** → Promote and remove flag after 48h
8. **Rollback never tested** → Test rollback weekly in staging
9. **A/B test ignores segments** → Analyze by segment, consider multi-model routing
10. **Production drift after 3 months** → Monitor continuously, retrain monthly

**Common themes:**
- **Baseline required:** Can't detect regressions without baseline
- **Statistical rigor:** Sample size calculations, significance testing
- **Thresholds enforced:** Latency, error rate, agreement rate
- **Gradual progression:** Never skip stages (shadow → canary → A/B → full)
- **Continuous monitoring:** Drift detection, performance tracking
- **Tested procedures:** Rollback, retraining, monitoring tested regularly

**Key insights:**
- **Deployment is a process, not an event:** Shadow → Canary → A/B → Full takes 5-7 days
- **Metrics matter:** Error rate, latency, agreement rate all critical
- **Rollback is infrastructure:** Must be instant, automated, tested
- **Models degrade:** Drift happens, retraining required monthly
- **Segments differ:** Aggregate metrics hide important patterns

---

## Complete Deployment Workflow

**Full production deployment workflow:**

```
Day 0: Baseline collection (24-48h)
├─ Collect metrics from current model
├─ Establish thresholds (error rate, latency, etc.)
└─ Document baseline for comparison

Day 1: Shadow mode (24-48h)
├─ Run new model alongside old (0% user impact)
├─ Compare outputs (agreement rate > 85%)
├─ Validate latency (< 1.5× baseline)
└─ Decision: Promote to canary or optimize

Day 2: Canary 5% (2-4h)
├─ Serve to 5% of users
├─ Monitor metrics every 5 minutes
├─ Auto-rollback if degraded
└─ Decision: Expand to 25% or rollback

Day 2: Canary 25% (6-12h)
├─ Serve to 25% of users
├─ Monitor metrics every 10 minutes
├─ Auto-rollback if degraded
└─ Decision: Expand to A/B test or rollback

Day 3: A/B test 50/50 (24-48h)
├─ Split traffic evenly
├─ Calculate statistical significance
├─ Measure effect size
├─ Analyze by segment
└─ Decision: Deploy 100% or rollback

Day 4-5: Full rollout 100% (48h monitoring)
├─ Deploy to all users
├─ Monitor for regressions
├─ Keep old model warm (instant rollback)
└─ Decision: Promote to production or rollback

Day 5-7: Promotion
├─ Promote new model to production (blue)
├─ Remove feature flags
├─ Unload old model
├─ Document deployment
└─ Set up monitoring for drift

Ongoing: Continuous monitoring
├─ Track metrics daily
├─ Detect drift weekly
├─ Retrain monthly
└─ Redeploy via same workflow
```

**Total timeline:** 5-7 days from baseline to full production.

**Critical success factors:**
1. ✓ Baseline established before deployment
2. ✓ Statistical rigor in A/B testing
3. ✓ Automated monitoring and rollback
4. ✓ Gradual progression (never skip stages)
5. ✓ Segment analysis for heterogeneous users
6. ✓ Continuous drift monitoring
7. ✓ Monthly retraining cadence
8. ✓ Tested rollback procedures
9. ✓ Feature flag lifecycle management
10. ✓ Documentation and runbooks

---

## Final Recommendations

**For AI Model Deployment:**

1. **Start with shadow mode:** Validate behavior before user impact
2. **Use gradual rollout:** Shadow → 5% → 25% → 50% → 100%
3. **Monitor automatically:** Metrics checked every 5 minutes
4. **Rollback instantly:** < 30 seconds via traffic switching
5. **Test statistically:** Calculate sample size, test significance
6. **Analyze segments:** Aggregate metrics hide patterns
7. **Retrain continuously:** Monthly retraining for drift
8. **Test rollback:** Weekly in staging
9. **Document everything:** Runbooks for deployment and rollback
10. **Promote and clean up:** Remove feature flags after 48h at 100%

**Deployment anti-patterns to avoid:**
- ❌ Instant 100% deployment
- ❌ A/B test with insufficient sample size
- ❌ Ignoring latency regressions
- ❌ Shadow mode without output comparison
- ❌ Skipping canary stages
- ❌ Untested rollback procedures
- ❌ Feature flags as permanent architecture
- ❌ Ignoring data drift
- ❌ Aggregate-only metrics (no segments)
- ❌ Deploy-and-forget (no continuous monitoring)

**Remember:** Safe deployment is systematic, gradual, monitored, and reversible. Take the time to do it right—your users will thank you.

---

## Conclusion

Deployment is not just pushing code—it's a systematic process of validation, monitoring, and risk mitigation. The patterns in this skill (A/B testing, canary deployments, shadow mode, blue-green with feature flags) provide the infrastructure for safe, confident deployments.

Master these patterns, avoid the anti-patterns, and you'll deploy AI models to production with confidence and safety.
