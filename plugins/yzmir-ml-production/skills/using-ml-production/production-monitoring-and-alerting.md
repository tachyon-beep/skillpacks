
# Production Monitoring and Alerting

## Overview

Comprehensive production monitoring and alerting for ML systems. Implements performance metrics (RED), model quality tracking, drift detection, dashboard design, alert rules, and SLAs/SLOs. Extends to LLM-specific observability for systems that serve generative models, RAG pipelines, and agentic workloads.

**Core Principle**: You can't improve what you don't measure. Monitoring is non-negotiable for production ML — deploy with observability or don't deploy.

## Section 1: Performance Metrics (RED Metrics)

### Foundation: Rate, Errors, Duration

**Every ML service must track:**

```python
from prometheus_client import Counter, Histogram, Gauge
import time
import functools

# REQUEST RATE (R)
REQUEST_COUNT = Counter(
    'ml_requests_total',
    'Total ML inference requests',
    ['model_name', 'endpoint', 'model_version']
)

# ERROR RATE (E)
ERROR_COUNT = Counter(
    'ml_errors_total',
    'Total ML inference errors',
    ['model_name', 'endpoint', 'error_type']
)

# DURATION (D) - Latency
REQUEST_LATENCY = Histogram(
    'ml_request_duration_seconds',
    'ML inference request latency',
    ['model_name', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Customize for your SLO
)

# Additional: In-flight requests (for load monitoring)
IN_PROGRESS = Gauge(
    'ml_requests_in_progress',
    'ML inference requests currently being processed',
    ['model_name']
)

def monitor_ml_endpoint(model_name: str, endpoint: str):
    """Decorator to monitor any ML endpoint"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(
                model_name=model_name,
                endpoint=endpoint,
                model_version=get_model_version()
            ).inc()

            IN_PROGRESS.labels(model_name=model_name).inc()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                REQUEST_LATENCY.labels(
                    model_name=model_name,
                    endpoint=endpoint
                ).observe(time.time() - start_time)
                return result

            except Exception as e:
                ERROR_COUNT.labels(
                    model_name=model_name,
                    endpoint=endpoint,
                    error_type=type(e).__name__
                ).inc()
                raise

            finally:
                IN_PROGRESS.labels(model_name=model_name).dec()

        return wrapper
    return decorator
```

The `prometheus_client` Python library exposes these metric primitives over an HTTP scrape endpoint that Prometheus pulls on its `scrape_interval`. See <https://prometheus.github.io/client_python/> and the Prometheus data-model docs at <https://prometheus.io/docs/concepts/data_model/>.

### Latency Percentiles (P50, P95, P99)

```python
# Prometheus calculates percentiles from a Histogram via histogram_quantile():
#   histogram_quantile(0.95, rate(ml_request_duration_seconds_bucket[5m]))
# Choose buckets that bracket your SLO; histogram_quantile interpolates within
# the bucket containing the target quantile. See:
#   https://prometheus.io/docs/practices/histograms/

import numpy as np
from collections import deque

class LatencyTracker:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)

    def record(self, latency_seconds):
        self.latencies.append(latency_seconds)

    def get_percentiles(self):
        if not self.latencies:
            return None
        arr = np.array(self.latencies)
        return {
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "mean": np.mean(arr),
            "max": np.max(arr),
        }
```

For high-cardinality or precise quantile work prefer Prometheus **native histograms** (sparse buckets, GA in Prometheus 2.40+) or a `Summary` type — see <https://prometheus.io/docs/concepts/metric_types/>.

### Throughput Tracking

```python
THROUGHPUT_GAUGE = Gauge(
    'ml_throughput_requests_per_second',
    'Current requests per second',
    ['model_name']
)

class ThroughputMonitor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.request_times = deque()

    def record_request(self):
        now = time.time()
        self.request_times.append(now)

        # Keep only last 60 seconds
        cutoff = now - 60
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()

        # Update gauge
        throughput = len(self.request_times) / 60.0
        THROUGHPUT_GAUGE.labels(model_name=self.model_name).set(throughput)
```


## Section 2: Model Quality Metrics

### Prediction Distribution Tracking

```python
from prometheus_client import Counter

PREDICTION_COUNT = Counter(
    'ml_predictions_by_class',
    'Total predictions by class label',
    ['model_name', 'predicted_class']
)

def track_prediction(model_name: str, prediction: str):
    PREDICTION_COUNT.labels(
        model_name=model_name,
        predicted_class=prediction,
    ).inc()
```

Plot `rate(ml_predictions_by_class[1h])` per class to detect prediction-distribution drift (sudden spikes in one class are often the earliest sign of upstream-data or model regressions).

### Confidence Distribution Tracking

```python
CONFIDENCE_HISTOGRAM = Histogram(
    'ml_prediction_confidence',
    'Model prediction confidence scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LOW_CONFIDENCE_COUNT = Counter(
    'ml_low_confidence_predictions',
    'Predictions below confidence threshold',
    ['model_name', 'threshold']
)

def track_confidence(model_name: str, confidence: float, threshold: float = 0.7):
    CONFIDENCE_HISTOGRAM.labels(model_name=model_name).observe(confidence)
    if confidence < threshold:
        LOW_CONFIDENCE_COUNT.labels(
            model_name=model_name,
            threshold=str(threshold),
        ).inc()
```

### Per-Segment Performance

```python
SEGMENT_ACCURACY_GAUGE = Gauge(
    'ml_accuracy_by_segment',
    'Model accuracy for different data segments',
    ['model_name', 'segment']
)

class SegmentPerformanceTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.segments = {}  # segment -> {"correct": X, "total": Y}

    def record_prediction(self, segment: str, is_correct: bool):
        if segment not in self.segments:
            self.segments[segment] = {"correct": 0, "total": 0}
        self.segments[segment]["total"] += 1
        if is_correct:
            self.segments[segment]["correct"] += 1
        accuracy = self.segments[segment]["correct"] / self.segments[segment]["total"]
        SEGMENT_ACCURACY_GAUGE.labels(
            model_name=self.model_name,
            segment=segment,
        ).set(accuracy)
```

### Ground Truth Sampling

```python
import random
from typing import Optional

class GroundTruthSampler:
    def __init__(self, model_name: str, sampling_rate: float = 0.1):
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.predictions = []
        self.ground_truths = []

    def sample_prediction(self, request_id: str, prediction: dict) -> bool:
        if random.random() < self.sampling_rate:
            self.predictions.append({
                "request_id": request_id,
                "prediction": prediction,
                "timestamp": time.time(),
            })
            send_to_review_queue(request_id, prediction)  # Label Studio, Argilla, etc.
            return True
        return False

    def add_ground_truth(self, request_id: str, ground_truth: str):
        self.ground_truths.append({
            "request_id": request_id,
            "ground_truth": ground_truth,
            "timestamp": time.time(),
        })
        if len(self.ground_truths) >= 100:
            self.calculate_accuracy()

    def calculate_accuracy(self):
        recent = self.ground_truths[-100:]
        pred_map = {p["request_id"]: p["prediction"] for p in self.predictions}
        correct = sum(1 for gt in recent if pred_map.get(gt["request_id"]) == gt["ground_truth"])
        accuracy = correct / len(recent)
        SEGMENT_ACCURACY_GAUGE.labels(
            model_name=self.model_name,
            segment="ground_truth_sample",
        ).set(accuracy)
        return accuracy
```

Common review-queue back-ends include **Label Studio** (<https://labelstud.io>) and **Argilla** (<https://argilla.io>), both of which integrate with model-prediction streams and produce labeled datasets that can feed back into retraining pipelines.


## Section 3: Data Drift Detection

### Kolmogorov–Smirnov Test (Distribution Comparison)

```python
from scipy.stats import ks_2samp
import numpy as np
from prometheus_client import Gauge, Counter

DRIFT_SCORE_GAUGE = Gauge(
    'ml_data_drift_score',
    'KS test D-statistic for data drift',
    ['model_name', 'feature_name']
)

DRIFT_ALERT = Counter(
    'ml_data_drift_alerts',
    'Data drift alerts triggered',
    ['model_name', 'feature_name', 'severity']
)

class DataDriftDetector:
    def __init__(self, model_name: str, reference_data: dict, window_size: int = 1000):
        self.model_name = model_name
        self.reference_data = reference_data
        self.window_size = window_size
        self.current_window = {feature: [] for feature in reference_data.keys()}
        self.thresholds = {"info": 0.1, "warning": 0.15, "critical": 0.25}

    def add_sample(self, features: dict):
        for feature_name, value in features.items():
            if feature_name in self.current_window:
                self.current_window[feature_name].append(value)
        first_feature = next(iter(self.current_window))
        if len(self.current_window[first_feature]) >= self.window_size:
            self.check_drift()
            self.current_window = {f: [] for f in self.reference_data.keys()}

    def check_drift(self):
        for feature_name, reference in self.reference_data.items():
            current = np.array(self.current_window[feature_name])
            statistic, p_value = ks_2samp(reference, current)
            DRIFT_SCORE_GAUGE.labels(
                model_name=self.model_name,
                feature_name=feature_name,
            ).set(statistic)
            severity = self._get_severity(statistic)
            if severity:
                DRIFT_ALERT.labels(
                    model_name=self.model_name,
                    feature_name=feature_name,
                    severity=severity,
                ).inc()

    def _get_severity(self, ks_statistic: float) -> Optional[str]:
        if ks_statistic >= self.thresholds["critical"]:
            return "critical"
        if ks_statistic >= self.thresholds["warning"]:
            return "warning"
        if ks_statistic >= self.thresholds["info"]:
            return "info"
        return None
```

KS interpretation: statistic D < 0.1 ≈ no drift; 0.1–0.15 slight; 0.15–0.25 moderate; > 0.25 severe. P-value alone is misleading at high N (any tiny difference becomes "significant"); track effect size (D) and complement with KL divergence or Wasserstein distance for richer signal.

### Population Stability Index (PSI) for Concept Drift

```python
PSI_GAUGE = Gauge(
    'ml_concept_drift_psi',
    'Population Stability Index for concept drift',
    ['model_name'],
)

class ConceptDriftDetector:
    def __init__(self, model_name: str, num_bins: int = 10):
        self.model_name = model_name
        self.num_bins = num_bins
        self.baseline_distribution = None
        self.current_predictions = []
        self.window_size = 1000
        self.thresholds = {"info": 0.1, "warning": 0.2, "critical": 0.25}

    def track_prediction(self, prediction: float):
        self.current_predictions.append(prediction)
        if len(self.current_predictions) >= self.window_size:
            self.check_concept_drift()
            self.current_predictions = []

    def _calculate_distribution(self, values: list) -> np.ndarray:
        hist, _ = np.histogram(values, bins=self.num_bins, range=(0, 1))
        return hist / max(len(values), 1)

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray) -> float:
        expected = np.where(expected == 0, 1e-4, expected)
        actual = np.where(actual == 0, 1e-4, actual)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    def check_concept_drift(self):
        if self.baseline_distribution is None:
            self.baseline_distribution = self._calculate_distribution(self.current_predictions)
            return None
        current = self._calculate_distribution(self.current_predictions)
        psi = self.calculate_psi(self.baseline_distribution, current)
        PSI_GAUGE.labels(model_name=self.model_name).set(psi)
        return psi
```

PSI rules of thumb: < 0.1 stable; 0.1–0.2 minor shift; > 0.2 material shift, investigate. Pair PSI with KL divergence for symmetric distributional comparison.

### Drift-Detection Libraries

For production-grade drift detection, prefer well-tested libraries over hand-rolled tests:

- **Evidently** — Python library + dashboard for data, target, and prediction drift. <https://docs.evidentlyai.com>
- **NannyML** — post-deployment performance estimation when ground truth is delayed (CBPE, DLE algorithms). <https://nannyml.readthedocs.io>
- **Alibi Detect** — drift, outlier, and adversarial detection (Seldon). <https://docs.seldon.io/projects/alibi-detect>
- **Deepchecks** — train/serve and continuous validation suites. <https://docs.deepchecks.com>
- **TorchDrift** — drift detection for PyTorch tensor inputs. <https://torchdrift.org>


## Section 4: Dashboard Design

### Tiered Dashboard Structure

```yaml
Dashboard Hierarchy:

Page 1 - SYSTEM HEALTH (single pane of glass):
  Purpose: Answer "Is the system healthy?" in 5 seconds
  Metrics:
    - Request rate (current vs normal)
    - Error rate (% and count)
    - Latency P95 (current vs SLO)
    - Model accuracy (ground truth sample)
  Layout: 4 large panels, color-coded (green/yellow/red)

Page 2 - MODEL QUALITY:
  Purpose: Deep dive into model performance
  Metrics:
    - Prediction distribution (over time)
    - Confidence distribution (histogram)
    - Per-segment accuracy (if applicable)
    - Ground truth accuracy (rolling window)

Page 3 - DRIFT DETECTION:
  Purpose: Detect model degradation early
  Metrics:
    - Data drift (KS test per feature)
    - Concept drift (PSI over time)
    - Feature distributions (current vs baseline)

Page 4 - RESOURCES (only check when alerted):
  Purpose: Debug resource issues
  Metrics:
    - CPU / memory / GPU utilization, disk I/O, network saturation
```

### Grafana Dashboard Example

Grafana (<https://grafana.com/docs/grafana/latest/>) consumes Prometheus as a data source and supports dashboards-as-code via JSON or the Terraform/Tanka providers. Key panel ideas:

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Latency P95",
        "type": "timeseries",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(ml_request_duration_seconds_bucket[5m]))"
        }],
        "fieldConfig": {"defaults": {"thresholds": {"steps": [
          {"color": "green",  "value": null},
          {"color": "yellow", "value": 0.5},
          {"color": "red",    "value": 1.0}
        ]}}}
      },
      {
        "title": "Data Drift (KS Statistic)",
        "type": "timeseries",
        "targets": [{
          "expr": "ml_data_drift_score",
          "legendFormat": "{{feature_name}}"
        }]
      }
    ]
  }
}
```

Pair Grafana with **Loki** for logs and **Tempo** for traces (all part of the Grafana LGTM stack — see <https://grafana.com/oss/>) to correlate metrics, logs, and traces against the same trace IDs.


## Section 5: Alert Rules (Actionable, Not Noisy)

### Severity-Based Alerting

```yaml
CRITICAL (page on-call):
  - Error rate > 5% for 5 minutes
  - Latency P95 > 2× SLO for 10 minutes
  - Service down (health check fails)
  - Model accuracy < 60% on ground-truth sample
  Response: 15 minutes; escalate if no ack

WARNING (Slack notify):
  - Error rate > 2% for 10 minutes
  - Latency P95 > 1.5× SLO for 15 minutes
  - Data drift KS > 0.15 for 1 hour
  - Low-confidence predictions > 20%
  Response: 1 hour

INFO (log/dashboard only):
  - Error rate > 1%
  - Latency increasing trend
  - KS > 0.10 (slight drift)
  - PSI > 0.10
```

### Prometheus Alert Rules

```yaml
groups:
  - name: ml_model_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          rate(ml_errors_total[5m]) / rate(ml_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "High error rate detected"
          description: "{{ $labels.model_name }} error rate is {{ $value | humanizePercentage }}"
          runbook: "https://wiki/runbooks/ml-high-error-rate"

      - alert: HighLatencyP95
        expr: |
          histogram_quantile(0.95, rate(ml_request_duration_seconds_bucket[5m])) > 1.0
        for: 10m
        labels:
          severity: critical

      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.15
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected on {{ $labels.feature_name }}"
          description: "KS statistic {{ $value }} above moderate threshold (0.15)"

      - alert: LowModelAccuracy
        expr: ml_accuracy_by_segment{segment="ground_truth_sample"} < 0.70
        for: 30m
        labels:
          severity: critical
```

See Prometheus alerting rule syntax at <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>.

### Alert Routing (Alertmanager)

```yaml
route:
  group_by: ['model_name', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match: { severity: critical }
      receiver: pagerduty
      continue: true
    - match: { severity: warning }
      receiver: slack_warnings
    - match: { severity: info }
      receiver: slack_info

receivers:
  - name: pagerduty
    pagerduty_configs: [{ service_key: <KEY> }]
  - name: slack_warnings
    slack_configs:
      - api_url: <WEBHOOK>
        channel: '#ml-alerts-warnings'
```

Alertmanager docs: <https://prometheus.io/docs/alerting/latest/alertmanager/>. For incident response and on-call rotations: PagerDuty (<https://www.pagerduty.com>), Opsgenie (<https://www.atlassian.com/software/opsgenie>), or open-source **Grafana OnCall** (<https://grafana.com/products/oncall/>).


## Section 6: SLAs and SLOs for ML Systems

### Defining Service Level Objectives

```yaml
Service: <Model Name>     Owner: <Team>     Version: <semver>

LATENCY:       P50 < 100ms, P95 < 500ms, P99 < 1000ms (95% compliance / month)
AVAILABILITY:  Uptime > 99.5% (≈ 3.6 hours downtime/month allowed)
ERROR RATE:    < 1% of requests fail
ACCURACY:      > 85% on ground-truth sample (rolling 1000)
THROUGHPUT:    Sustain 1000 req/s without degradation
COST:          < $0.05 per 1000 requests
```

Use Google's SRE workbook framing — error budgets, burn rates, multi-window/multi-burn alerts: <https://sre.google/workbook/alerting-on-slos/>.

### SLO Compliance Tracking

```python
SLO_COMPLIANCE_GAUGE = Gauge(
    'ml_slo_compliance_percentage',
    'SLO compliance percentage',
    ['model_name', 'slo_type']
)

class SLOTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.slos = {
            "latency_p95": {"target": 0.5,  "compliance": 0.95},
            "error_rate":  {"target": 0.01, "compliance": 0.95},
            "accuracy":    {"target": 0.85, "compliance": 0.95},
            "availability":{"target": 0.995,"compliance": 1.00},
        }
        self.measurements = {slo: [] for slo in self.slos}

    def record_measurement(self, slo_type: str, value: float):
        self.measurements[slo_type].append({
            "value": value,
            "timestamp": time.time(),
            "compliant": self._is_compliant(slo_type, value),
        })
        cutoff = time.time() - 30 * 24 * 3600
        self.measurements[slo_type] = [m for m in self.measurements[slo_type] if m["timestamp"] > cutoff]
        compliance = self.calculate_compliance(slo_type)
        SLO_COMPLIANCE_GAUGE.labels(model_name=self.model_name, slo_type=slo_type).set(compliance)

    def _is_compliant(self, slo_type: str, value: float) -> bool:
        target = self.slos[slo_type]["target"]
        return value <= target if slo_type in ("latency_p95", "error_rate") else value >= target

    def calculate_compliance(self, slo_type: str) -> float:
        if not self.measurements[slo_type]:
            return 0.0
        return sum(1 for m in self.measurements[slo_type] if m["compliant"]) / len(self.measurements[slo_type])
```

For a richer SLO toolchain, see **Sloth** (<https://sloth.dev>) and **Pyrra** (<https://github.com/pyrra-dev/pyrra>) — both generate Prometheus alerts and recording rules from declarative SLO specs.


## Section 7: Monitoring Stack (Prometheus + Grafana + OpenTelemetry)

### Reference Stack

```yaml
# docker-compose.yml
services:
  ml-service:
    build: .
    ports: ["8000:8000", "8001:8001"]   # 8001 = /metrics
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus_rules.yml:/etc/prometheus/rules.yml
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otelcol/config.yaml"]
```

```python
# Minimal FastAPI service exposing /metrics
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('ml_latency_seconds', 'Request latency', ['endpoint'])

@app.post("/predict")
@REQUEST_LATENCY.labels(endpoint="/predict").time()
def predict(text: str):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    return {"prediction": model.predict(text)}

app.mount("/metrics", make_asgi_app())
```

**OpenTelemetry** (<https://opentelemetry.io>) is the vendor-neutral standard for traces, metrics, and logs. The OpenTelemetry Collector exports to Prometheus, Tempo, Loki, Datadog, Honeycomb, etc., via a single config — instrument once, switch back-ends without code changes.


## Section 8: LLM Observability

LLM applications (chat assistants, RAG systems, agents) generate signals that traditional ML monitoring stacks don't capture: per-token cost, streaming latency, hallucination rates, prompt-injection attempts, tool-call success, retrieval relevance. This section covers the LLM-observability ecosystem and the new metrics it surfaces.

Cross-references: see `yzmir-llm-specialist/llm-evaluation-metrics.md` for the offline eval methodology that production monitoring runs continuously, and `yzmir-llm-specialist/llm-safety-alignment.md` for production safety signals (refusals, jailbreak attempts, policy violations).

### What's Different About LLM Monitoring

| Traditional ML | LLM Application |
|----|----|
| Single inference call | Multi-turn conversation, tool calls, retrieval steps |
| Per-request latency | Time-to-first-token (TTFT) + tokens/sec streaming |
| Error rate | Refusal rate, hallucination rate, format violations |
| Predictions in fixed schema | Free-form text (needs eval-as-judge to score) |
| Cost = compute time | Cost = input_tokens × $/M + output_tokens × $/M |
| Drift = feature distribution | Drift = prompt distribution, retrieval relevance, tool-call mix |
| Adversarial input rare | Prompt injection, jailbreaks are constant background traffic |

### Core LLM Signals to Track

**Cost and usage:**
- Input/output tokens per request, per user, per feature, per provider tier
- Cost per request (computed from provider price sheet for the model tier in use)
- Cost per active user, per conversation, per resolved task
- Cache-hit rate (for prompt caching — see `yzmir-llm-specialist/context-engineering-and-prompt-caching.md`)

**Latency:**
- Time-to-first-token (TTFT) — user-perceived "is it working?" latency
- Tokens-per-second during streaming
- End-to-end latency including all tool calls and retrieval hops
- Queue/throttle wait time at provider rate limits

**Quality:**
- LLM-as-judge scores on sampled production traffic (cross-ref `llm-evaluation-metrics.md`)
- Hallucination/groundedness rate for RAG (citations resolve, claims supported by retrieved docs)
- Refusal rate (legitimate refusals vs over-refusals)
- Format/schema compliance (JSON validity, function-call argument validity)
- Tool-call success rate and tool-error taxonomy
- Retrieval metrics: recall@k, MRR, hit-rate on labeled query sets

**Safety:**
- Detected prompt-injection attempts (cross-ref `llm-safety-alignment.md`)
- Policy-violation detections (PII, toxicity, off-topic)
- Jailbreak success rate (system-prompt leakage, persona override)

### Tooling Landscape

**Open-source / self-hostable:**

- **Arize Phoenix** — open-source LLM tracing, evaluation, and experiment tracking with OpenTelemetry-native traces. Notebooks-first DX, runs locally or self-hosted. Docs: <https://docs.arize.com/phoenix>. Repo: <https://github.com/Arize-ai/phoenix>.
- **Langfuse** — open-source LLM observability + prompt versioning + dataset/eval management; SDKs for Python and JS, self-host via Docker or use Langfuse Cloud. <https://langfuse.com> and <https://langfuse.com/docs>.
- **Helicone** — proxy-based observability (point your OpenAI-compatible client at Helicone's endpoint and it logs every request) with caching, retries, and per-user cost tracking. Open-source core, also offered as cloud. <https://helicone.ai> and <https://docs.helicone.ai>.
- **OpenLLMetry** (Traceloop) — OpenTelemetry-native instrumentation library that emits LLM-call spans following the GenAI semantic conventions. <https://www.traceloop.com/openllmetry>.
- **Opik** (Comet) — open-source LLM-evaluation and tracing tool, complements Comet's broader experiment tracking. <https://www.comet.com/site/products/opik/> and <https://github.com/comet-ml/opik>.
- **DeepEval** — open-source LLM eval library with metrics like answer-relevance, faithfulness, contextual-precision; integrates with CI. <https://docs.confident-ai.com>.

**Commercial / managed:**

- **Arize AI** — enterprise platform combining traditional ML monitoring with LLM observability; ships Phoenix as the OSS path. <https://arize.com>.
- **WhyLabs** — drift, data-quality, and LLM observability built on the open-source `whylogs` profiler. <https://whylabs.ai> and <https://whylogs.readthedocs.io>.
- **Fiddler AI** — model monitoring, explainability, and LLM observability with built-in safety/hallucination metrics. <https://www.fiddler.ai>.
- **Aporia** — ML and LLM observability with guardrails (PII, prompt-injection, hallucination detection inline). <https://www.aporia.com>.
- **Datadog LLM Observability** — LLM tracing and evaluation as a module within Datadog APM. <https://docs.datadoghq.com/llm_observability/>.
- **New Relic AI Monitoring** — APM-integrated LLM observability. <https://newrelic.com/platform/ai-monitoring>.
- **Comet MPM (Model Production Monitoring)** — drift, quality, and integrity for ML and LLM models, integrates with Comet Experiments. <https://www.comet.com/site/products/model-production-monitoring/>.
- **Deepchecks Monitoring** — continuous testing and monitoring for ML and LLM systems. <https://www.deepchecks.com/llm-evaluation-monitoring/>.
- **Galileo** — LLM evaluation, observability, and guardrails with proprietary "Luna" eval models. <https://www.galileo.ai>.
- **Braintrust** — LLM evals + tracing + prompt playground. <https://www.braintrust.dev>.
- **LangSmith** (LangChain) — tracing, eval, and prompt management; deepest integration with LangChain/LangGraph but works with any LLM provider via SDK. <https://docs.smith.langchain.com>.

### OpenTelemetry GenAI Semantic Conventions

OpenTelemetry has emerging — and still evolving — semantic conventions for GenAI spans and metrics. Following them lets one instrumentation library export to any OTel-compatible back-end (Phoenix, Langfuse, Datadog, Grafana Tempo, Honeycomb, etc.). See the live spec at <https://opentelemetry.io/docs/specs/semconv/gen-ai/> and the metrics conventions at <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/>.

Stable-ish span attributes (still subject to change — pin OTel SDK versions and re-check quarterly):

- `gen_ai.system` (e.g. `"openai"`, `"anthropic"`, `"vertex_ai"`)
- `gen_ai.request.model`, `gen_ai.response.model`
- `gen_ai.request.temperature`, `gen_ai.request.top_p`, `gen_ai.request.max_tokens`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.response.finish_reasons`
- Operation events: `gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message`, `gen_ai.choice`

Standard metric instruments include `gen_ai.client.token.usage` (histogram) and `gen_ai.client.operation.duration` (histogram). The benefit: a single OTel-instrumented service emits traces consumable by Phoenix, metrics scrapeable by Prometheus, and logs shippable to Loki — with no per-back-end glue.

### Worked Example: LLM Service With Phoenix + Prometheus

```python
# requirements: arize-phoenix, openinference-instrumentation-openai,
#               opentelemetry-sdk, prometheus_client, fastapi
import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI

# 1) Start (or connect to) Phoenix collector — it speaks OTLP.
session = px.launch_app()  # or run as a separate service
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=session.url + "/v1/traces"))
)
trace.set_tracer_provider(tracer_provider)

# 2) Auto-instrument your LLM client (here: OpenAI-compatible).
OpenAIInstrumentor().instrument()

# 3) Add LLM-specific Prometheus metrics alongside RED metrics.
TOKENS_INPUT  = Counter('llm_input_tokens_total',  'Input tokens',  ['model_tier', 'feature'])
TOKENS_OUTPUT = Counter('llm_output_tokens_total', 'Output tokens', ['model_tier', 'feature'])
COST_USD      = Counter('llm_cost_usd_total',      'Estimated USD cost', ['model_tier', 'feature'])
TTFT          = Histogram('llm_time_to_first_token_seconds', 'TTFT', ['model_tier', 'feature'],
                          buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10])
TOOL_CALLS    = Counter('llm_tool_calls_total', 'Tool invocations', ['tool', 'status'])
JUDGE_SCORE   = Histogram('llm_judge_score', 'LLM-as-judge score (0-1)',
                          ['feature', 'metric'], buckets=[0.1*i for i in range(11)])
INJECTION_DETECTED = Counter('llm_prompt_injection_detected_total',
                             'Detected prompt-injection attempts', ['feature'])

app = FastAPI()
app.mount("/metrics", make_asgi_app())
```

Key practices:

- **Sample, don't store everything.** Full prompt/response traces are expensive at scale; sample 5–20% for traces, log aggregates always. Phoenix and Langfuse both support sampling.
- **Redact PII before exporting.** Use the OTel collector's `attributes` and `redaction` processors, or your observability provider's PII-scrubbing config.
- **Run eval continuously, not just in CI.** Score a sampled slice of production traffic with an LLM-as-judge model and emit those scores to Prometheus / your observability tool. See `yzmir-llm-specialist/llm-evaluation-metrics.md` for judge-prompt design and metric definitions.
- **Track tool-call taxonomy.** For agent systems, the most common failure mode is tool-call malformation or unrecoverable tool errors — instrument every tool invocation with `{tool, status, latency, error_type}`.

### LLM Alert Rules

```yaml
groups:
  - name: llm_alerts
    rules:
      - alert: LLMCostBurnHigh
        expr: rate(llm_cost_usd_total[1h]) > 50  # $/hour
        for: 15m
        labels: { severity: warning }
      - alert: LLMTimeToFirstTokenHigh
        expr: histogram_quantile(0.95, rate(llm_time_to_first_token_seconds_bucket[5m])) > 3
        for: 10m
        labels: { severity: warning }
      - alert: LLMJudgeScoreDropped
        expr: avg_over_time(llm_judge_score{metric="faithfulness"}[1h]) < 0.7
        for: 30m
        labels: { severity: critical }
      - alert: LLMInjectionAttemptsSpiking
        expr: rate(llm_prompt_injection_detected_total[15m]) > 5 * rate(llm_prompt_injection_detected_total[24h] offset 1d)
        for: 15m
        labels: { severity: warning }
      - alert: ToolCallSuccessRateLow
        expr: |
          sum(rate(llm_tool_calls_total{status="success"}[5m])) by (tool)
          / sum(rate(llm_tool_calls_total[5m])) by (tool) < 0.9
        for: 15m
        labels: { severity: warning }
```

### LLM Monitoring Selection Matrix

| Need | Recommended path |
|----|----|
| OSS, OTel-native, notebooks-first | Arize Phoenix + OpenLLMetry |
| OSS with prompt versioning + datasets | Langfuse (self-hosted) |
| Lowest-friction proxy / cost dashboard | Helicone |
| Already on Datadog / New Relic | Their LLM module — single pane with APM |
| Enterprise with safety/hallucination guardrails | Fiddler, Aporia, Galileo |
| Deep LangChain/LangGraph users | LangSmith |
| Multi-tenant SaaS, per-user cost attribution | Helicone or Langfuse with user-id tagging |
| Need eval-as-CI gate alongside production tracing | Braintrust, Opik, DeepEval |

The pattern that ages best: **instrument with OpenTelemetry (GenAI semconv) + a thin vendor-specific instrumentation library (OpenInference, OpenLLMetry, Langfuse SDK)**. The trace data then flows to whichever back-end you choose now and can be redirected later without code changes.


## Section 9: Complete Example (End-to-End)

```python
# complete_monitoring.py — RED + model-quality + drift + LLM signals
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time

app = FastAPI()

# RED
REQUEST_COUNT = Counter('ml_requests_total', 'Requests', ['endpoint', 'model_version'])
ERROR_COUNT   = Counter('ml_errors_total',   'Errors',   ['error_type'])
LATENCY       = Histogram('ml_latency_seconds', 'Latency', ['endpoint'],
                          buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0])

# Model quality
PRED_BY_CLASS = Counter('ml_predictions_by_class', 'Predictions', ['class'])
CONFIDENCE    = Histogram('ml_prediction_confidence', 'Confidence',
                          buckets=[i/10 for i in range(11)])
ACCURACY      = Gauge('ml_accuracy_ground_truth', 'Sampled accuracy')

# Drift
DRIFT_KS = Gauge('ml_data_drift_ks',     'KS per feature', ['feature'])
DRIFT_PSI = Gauge('ml_concept_drift_psi','PSI on predictions')

# LLM (only emit when serving an LLM endpoint)
LLM_TOKENS_OUT = Counter('llm_output_tokens_total', 'Output tokens', ['model_tier'])
LLM_COST       = Counter('llm_cost_usd_total',       'Cost USD',     ['model_tier'])
LLM_TTFT       = Histogram('llm_time_to_first_token_seconds', 'TTFT')

@app.post("/predict")
def predict(text: str):
    start = time.time()
    try:
        result = {"label": "positive", "confidence": 0.92}
        LATENCY.labels(endpoint="/predict").observe(time.time() - start)
        REQUEST_COUNT.labels(endpoint="/predict", model_version="v1.0").inc()
        PRED_BY_CLASS.labels(class=result["label"]).inc()
        CONFIDENCE.observe(result["confidence"])
        return result
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise HTTPException(500, str(e))

app.mount("/metrics", make_asgi_app())
```


## Key Takeaways

1. **Monitoring is mandatory** — instrument before deployment, not after the first incident.
2. **RED metrics first** — Rate, Errors, Duration for every service.
3. **Model quality is its own pillar** — predictions, confidence, ground-truth-sampled accuracy.
4. **Drift detection prevents silent degradation** — KS / PSI / KL plus a library like Evidently or NannyML.
5. **Actionable alerts only** — severity tiers, runbook links, multi-burn-rate SLO alerts.
6. **SLOs define success quantitatively** — error budgets steer engineering priorities.
7. **LLM applications need a parallel observability layer** — TTFT, tokens, cost, judge scores, tool-call success, injection attempts; OpenTelemetry GenAI conventions are the durable substrate.
8. **Cross-pack integration**: production monitoring is where `llm-evaluation-metrics.md` (offline eval methodology) and `llm-safety-alignment.md` (safety signals) become continuous, not point-in-time.

Tooling and APIs current as of 2026-05; revisit quarterly.
