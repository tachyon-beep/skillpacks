---
name: production-monitoring-and-alerting
description: Monitor production models with performance metrics, drift detection, and alerting.
---

# Production Monitoring and Alerting

## Overview

Comprehensive production monitoring and alerting for ML systems. Implements performance metrics (RED), model quality tracking, drift detection, dashboard design, alert rules, and SLAs/SLOs.

**Core Principle**: You can't improve what you don't measure. Monitoring is non-negotiable for production ML - deploy with observability or don't deploy.

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

# Usage example
@monitor_ml_endpoint(model_name="sentiment_classifier", endpoint="/predict")
def predict_sentiment(text: str):
    result = model.predict(text)
    return result
```

### Latency Percentiles (P50, P95, P99)

```python
# Prometheus automatically calculates percentiles from Histogram
# Query in Prometheus:
#   P50: histogram_quantile(0.50, rate(ml_request_duration_seconds_bucket[5m]))
#   P95: histogram_quantile(0.95, rate(ml_request_duration_seconds_bucket[5m]))
#   P99: histogram_quantile(0.99, rate(ml_request_duration_seconds_bucket[5m]))

# For custom tracking:
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
            "max": np.max(arr)
        }
```

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

---

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
        predicted_class=prediction
    ).inc()

# Example: Sentiment classifier
result = model.predict("Great product!")  # Returns "positive"
track_prediction("sentiment_classifier", result)

# Dashboard query: Check if prediction distribution is shifting
# rate(ml_predictions_by_class{predicted_class="positive"}[1h])
```

### Confidence Distribution Tracking

```python
CONFIDENCE_HISTOGRAM = Histogram(
    'ml_prediction_confidence',
    'Model prediction confidence scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
            threshold=str(threshold)
        ).inc()

# Alert if low confidence predictions increase (model uncertainty rising)
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

        # Update gauge
        accuracy = self.segments[segment]["correct"] / self.segments[segment]["total"]
        SEGMENT_ACCURACY_GAUGE.labels(
            model_name=self.model_name,
            segment=segment
        ).set(accuracy)

# Example: E-commerce recommendations
tracker = SegmentPerformanceTracker("recommender")
tracker.record_prediction(segment="electronics", is_correct=True)
tracker.record_prediction(segment="clothing", is_correct=False)

# Alert if accuracy drops for specific segment (targeted debugging)
```

### Ground Truth Sampling

```python
import random
from typing import Optional

class GroundTruthSampler:
    def __init__(self, model_name: str, sampling_rate: float = 0.1):
        """
        sampling_rate: Fraction of predictions to send for human review (0.0-1.0)
        """
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.predictions = []
        self.ground_truths = []

    def sample_prediction(self, request_id: str, prediction: dict) -> bool:
        """
        Returns True if prediction should be sent for human review
        """
        if random.random() < self.sampling_rate:
            self.predictions.append({
                "request_id": request_id,
                "prediction": prediction,
                "timestamp": time.time()
            })
            # Send to review queue (e.g., Label Studio, human review dashboard)
            send_to_review_queue(request_id, prediction)
            return True
        return False

    def add_ground_truth(self, request_id: str, ground_truth: str):
        """Human reviewer provides true label"""
        self.ground_truths.append({
            "request_id": request_id,
            "ground_truth": ground_truth,
            "timestamp": time.time()
        })

        # Calculate rolling accuracy
        if len(self.ground_truths) >= 100:
            self.calculate_accuracy()

    def calculate_accuracy(self):
        """Calculate accuracy on last N samples"""
        recent = self.ground_truths[-100:]
        pred_map = {p["request_id"]: p["prediction"] for p in self.predictions}

        correct = sum(
            1 for gt in recent
            if pred_map.get(gt["request_id"]) == gt["ground_truth"]
        )

        accuracy = correct / len(recent)

        SEGMENT_ACCURACY_GAUGE.labels(
            model_name=self.model_name,
            segment="ground_truth_sample"
        ).set(accuracy)

        return accuracy

# Usage
sampler = GroundTruthSampler("sentiment_classifier", sampling_rate=0.1)

@app.post("/predict")
def predict(text: str):
    result = model.predict(text)
    request_id = generate_request_id()

    # Sample for human review
    sampler.sample_prediction(request_id, result)

    return {"request_id": request_id, "result": result}

# Later: Human reviewer provides label
@app.post("/feedback")
def feedback(request_id: str, true_label: str):
    sampler.add_ground_truth(request_id, true_label)
    return {"status": "recorded"}
```

---

## Section 3: Data Drift Detection

### Kolmogorov-Smirnov Test (Distribution Comparison)

```python
from scipy.stats import ks_2samp
import numpy as np
from prometheus_client import Gauge

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
        """
        reference_data: Dict of feature_name -> np.array of training data values
        window_size: Number of production samples before checking drift
        """
        self.model_name = model_name
        self.reference_data = reference_data
        self.window_size = window_size
        self.current_window = {feature: [] for feature in reference_data.keys()}

        # Drift thresholds
        self.thresholds = {
            "info": 0.1,      # Slight shift (log only)
            "warning": 0.15,  # Moderate shift (investigate)
            "critical": 0.25  # Severe shift (retrain needed)
        }

    def add_sample(self, features: dict):
        """Add new production sample"""
        for feature_name, value in features.items():
            if feature_name in self.current_window:
                self.current_window[feature_name].append(value)

        # Check drift when window full
        if len(self.current_window[list(self.current_window.keys())[0]]) >= self.window_size:
            self.check_drift()
            # Reset window
            self.current_window = {feature: [] for feature in self.reference_data.keys()}

    def check_drift(self):
        """Compare current window to reference using KS test"""
        results = {}

        for feature_name in self.reference_data.keys():
            reference = self.reference_data[feature_name]
            current = np.array(self.current_window[feature_name])

            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(reference, current)

            results[feature_name] = {
                "ks_statistic": statistic,
                "p_value": p_value
            }

            # Update Prometheus gauge
            DRIFT_SCORE_GAUGE.labels(
                model_name=self.model_name,
                feature_name=feature_name
            ).set(statistic)

            # Alert if drift detected
            severity = self._get_severity(statistic)
            if severity:
                DRIFT_ALERT.labels(
                    model_name=self.model_name,
                    feature_name=feature_name,
                    severity=severity
                ).inc()
                self._send_alert(feature_name, statistic, p_value, severity)

        return results

    def _get_severity(self, ks_statistic: float) -> Optional[str]:
        """Determine alert severity based on KS statistic"""
        if ks_statistic >= self.thresholds["critical"]:
            return "critical"
        elif ks_statistic >= self.thresholds["warning"]:
            return "warning"
        elif ks_statistic >= self.thresholds["info"]:
            return "info"
        return None

    def _send_alert(self, feature_name: str, ks_stat: float, p_value: float, severity: str):
        """Send drift alert to monitoring system"""
        message = f"""
DATA DRIFT DETECTED

Model: {self.model_name}
Feature: {feature_name}
Severity: {severity.upper()}

KS Statistic: {ks_stat:.3f}
P-value: {p_value:.4f}

Interpretation:
- KS < 0.1: No significant drift
- KS 0.1-0.15: Slight shift (monitor)
- KS 0.15-0.25: Moderate drift (investigate)
- KS > 0.25: Severe drift (retrain recommended)

Action:
1. Review recent input examples
2. Check for data source changes
3. Compare distributions visually
4. Consider retraining if accuracy dropping
        """
        send_alert_to_slack(message)  # Or PagerDuty, email, etc.

# Usage example
# Training data statistics
reference_features = {
    "text_length": np.random.normal(100, 20, 10000),  # Mean 100, std 20
    "sentiment_score": np.random.normal(0.5, 0.2, 10000),  # Mean 0.5, std 0.2
}

drift_detector = DataDriftDetector("sentiment_classifier", reference_features)

@app.post("/predict")
def predict(text: str):
    # Extract features
    features = {
        "text_length": len(text),
        "sentiment_score": get_sentiment_score(text)
    }

    # Track for drift detection
    drift_detector.add_sample(features)

    result = model.predict(text)
    return result
```

### Population Stability Index (PSI) for Concept Drift

```python
import numpy as np

PSI_GAUGE = Gauge(
    'ml_concept_drift_psi',
    'Population Stability Index for concept drift',
    ['model_name']
)

class ConceptDriftDetector:
    def __init__(self, model_name: str, num_bins: int = 10):
        """
        num_bins: Number of bins for PSI calculation
        """
        self.model_name = model_name
        self.num_bins = num_bins
        self.baseline_distribution = None
        self.current_predictions = []
        self.window_size = 1000

        # PSI thresholds
        self.thresholds = {
            "info": 0.1,      # Slight shift
            "warning": 0.2,   # Moderate shift (investigate)
            "critical": 0.25  # Severe shift (model behavior changed)
        }

    def set_baseline(self, predictions: list):
        """Set baseline prediction distribution (from first week of production)"""
        self.baseline_distribution = self._calculate_distribution(predictions)

    def track_prediction(self, prediction: float):
        """Track new prediction (probability or class)"""
        self.current_predictions.append(prediction)

        # Check concept drift when window full
        if len(self.current_predictions) >= self.window_size:
            self.check_concept_drift()
            self.current_predictions = []

    def _calculate_distribution(self, values: list) -> np.ndarray:
        """Calculate binned distribution"""
        hist, _ = np.histogram(values, bins=self.num_bins, range=(0, 1))
        # Convert to proportions
        return hist / len(values)

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Interpretation:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.2: Slight change (monitor)
        - PSI > 0.2: Significant change (investigate/retrain)
        """
        # Avoid division by zero
        expected = np.where(expected == 0, 0.0001, expected)
        actual = np.where(actual == 0, 0.0001, actual)

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return psi

    def check_concept_drift(self):
        """Check if model behavior has changed"""
        if self.baseline_distribution is None:
            # Set first window as baseline
            self.baseline_distribution = self._calculate_distribution(self.current_predictions)
            return None

        current_distribution = self._calculate_distribution(self.current_predictions)
        psi = self.calculate_psi(self.baseline_distribution, current_distribution)

        # Update Prometheus gauge
        PSI_GAUGE.labels(model_name=self.model_name).set(psi)

        # Alert if concept drift detected
        severity = self._get_severity(psi)
        if severity:
            self._send_alert(psi, severity)

        return psi

    def _get_severity(self, psi: float) -> Optional[str]:
        if psi >= self.thresholds["critical"]:
            return "critical"
        elif psi >= self.thresholds["warning"]:
            return "warning"
        elif psi >= self.thresholds["info"]:
            return "info"
        return None

    def _send_alert(self, psi: float, severity: str):
        message = f"""
CONCEPT DRIFT DETECTED

Model: {self.model_name}
Severity: {severity.upper()}

PSI: {psi:.3f}

Interpretation:
- PSI < 0.1: No significant change
- PSI 0.1-0.2: Slight change (model behavior shifting)
- PSI > 0.2: Significant change (model may need retraining)

Action:
1. Compare current vs baseline prediction distributions
2. Check if input distribution also changed (data drift?)
3. Validate accuracy on recent samples
4. Consider retraining if accuracy dropping
        """
        send_alert_to_slack(message)

# Usage
concept_drift_detector = ConceptDriftDetector("sentiment_classifier")

@app.post("/predict")
def predict(text: str):
    result = model.predict(text)
    confidence = result["confidence"]

    # Track prediction for concept drift
    concept_drift_detector.track_prediction(confidence)

    return result
```

---

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
  Layout: Time series + histograms

Page 3 - DRIFT DETECTION:
  Purpose: Detect model degradation early
  Metrics:
    - Data drift (KS test per feature)
    - Concept drift (PSI over time)
    - Feature distributions (current vs baseline)
  Layout: Time series + distribution comparisons

Page 4 - RESOURCES (only check when alerted):
  Purpose: Debug resource issues
  Metrics:
    - CPU utilization
    - Memory usage (RSS)
    - GPU utilization/memory (if applicable)
    - Disk I/O
  Layout: System resource graphs
```

### Grafana Dashboard Example (JSON)

```json
{
  "dashboard": {
    "title": "ML Model Monitoring - Sentiment Classifier",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(ml_requests_total{model_name=\"sentiment_classifier\"}[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph",
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 8}
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(ml_errors_total{model_name=\"sentiment_classifier\"}[5m]) / rate(ml_requests_total{model_name=\"sentiment_classifier\"}[5m])",
            "legendFormat": "Error %"
          }
        ],
        "type": "graph",
        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 8}
      },
      {
        "title": "Latency P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_request_duration_seconds_bucket{model_name=\"sentiment_classifier\"}[5m]))",
            "legendFormat": "P95"
          }
        ],
        "type": "graph",
        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8},
        "alert": {
          "conditions": [
            {
              "query": "A",
              "reducer": "avg",
              "evaluator": {"params": [0.5], "type": "gt"}
            }
          ],
          "message": "Latency P95 above 500ms SLO"
        }
      },
      {
        "title": "Prediction Distribution",
        "targets": [
          {
            "expr": "rate(ml_predictions_by_class{model_name=\"sentiment_classifier\"}[1h])",
            "legendFormat": "{{predicted_class}}"
          }
        ],
        "type": "graph",
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      },
      {
        "title": "Data Drift (KS Test)",
        "targets": [
          {
            "expr": "ml_data_drift_score{model_name=\"sentiment_classifier\"}",
            "legendFormat": "{{feature_name}}"
          }
        ],
        "type": "graph",
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
        "thresholds": [
          {"value": 0.15, "color": "yellow"},
          {"value": 0.25, "color": "red"}
        ]
      }
    ]
  }
}
```

---

## Section 5: Alert Rules (Actionable, Not Noisy)

### Severity-Based Alerting

```yaml
Alert Severity Levels:

CRITICAL (page immediately, wake up on-call):
  - Error rate > 5% for 5 minutes
  - Latency P95 > 2× SLO for 10 minutes
  - Service down (health check fails)
  - Model accuracy < 60% (catastrophic failure)
  Response time: 15 minutes
  Escalation: Page backup if no ack in 15 min

WARNING (notify, but don't wake up):
  - Error rate > 2% for 10 minutes
  - Latency P95 > 1.5× SLO for 15 minutes
  - Data drift KS > 0.15 (moderate)
  - Low confidence predictions > 20%
  Response time: 1 hour
  Escalation: Slack notification

INFO (log for review):
  - Error rate > 1%
  - Latency increasing trend
  - Data drift KS > 0.1 (slight)
  - Concept drift PSI > 0.1
  Response time: Next business day
  Escalation: Dashboard review
```

### Prometheus Alert Rules

```yaml
# prometheus_rules.yml

groups:
  - name: ml_model_alerts
    interval: 30s
    rules:

      # CRITICAL: High error rate
      - alert: HighErrorRate
        expr: |
          (
            rate(ml_errors_total[5m])
            /
            rate(ml_requests_total[5m])
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "High error rate detected"
          description: |
            Model {{ $labels.model_name }} error rate is {{ $value | humanizePercentage }}
            (threshold: 5%)

            RUNBOOK:
            1. Check recent error logs: kubectl logs -l app=ml-service --since=10m | grep ERROR
            2. Check model health: curl http://service/health
            3. Check recent deployments: kubectl rollout history deployment/ml-service
            4. If model OOM: kubectl scale --replicas=5 deployment/ml-service
            5. If persistent: Rollback to previous version

      # CRITICAL: High latency
      - alert: HighLatencyP95
        expr: |
          histogram_quantile(0.95,
            rate(ml_request_duration_seconds_bucket[5m])
          ) > 1.0
        for: 10m
        labels:
          severity: critical
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "Latency P95 above SLO"
          description: |
            Model {{ $labels.model_name }} latency P95 is {{ $value }}s
            (SLO: 0.5s, threshold: 1.0s = 2× SLO)

            RUNBOOK:
            1. Check current load: rate(ml_requests_total[5m])
            2. Check resource usage: CPU/memory/GPU utilization
            3. Check for slow requests: Check P99 latency
            4. Scale if needed: kubectl scale --replicas=10 deployment/ml-service
            5. Check downstream dependencies (database, cache, APIs)

      # WARNING: Moderate data drift
      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.15
        for: 1h
        labels:
          severity: warning
          model: "{{ $labels.model_name }}"
          feature: "{{ $labels.feature_name }}"
        annotations:
          summary: "Data drift detected"
          description: |
            Model {{ $labels.model_name }} feature {{ $labels.feature_name }}
            KS statistic: {{ $value }}
            (threshold: 0.15 = moderate drift)

            RUNBOOK:
            1. Compare current vs baseline distributions (Grafana dashboard)
            2. Check recent data source changes
            3. Review sample inputs for anomalies
            4. If drift severe (KS > 0.25): Plan retraining
            5. If accuracy dropping: Expedite retraining

      # WARNING: Concept drift
      - alert: ConceptDriftDetected
        expr: ml_concept_drift_psi > 0.2
        for: 1h
        labels:
          severity: warning
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "Concept drift detected"
          description: |
            Model {{ $labels.model_name }} PSI: {{ $value }}
            (threshold: 0.2 = significant shift)

            Model behavior is changing (same inputs → different outputs)

            RUNBOOK:
            1. Check prediction distribution changes (Grafana)
            2. Compare with data drift (correlated?)
            3. Validate accuracy on ground truth samples
            4. If accuracy < 75%: Retraining required
            5. Investigate root cause (seasonality, new patterns, etc.)

      # CRITICAL: Low accuracy
      - alert: LowModelAccuracy
        expr: ml_accuracy_by_segment{segment="ground_truth_sample"} < 0.70
        for: 30m
        labels:
          severity: critical
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "Model accuracy below threshold"
          description: |
            Model {{ $labels.model_name }} accuracy: {{ $value | humanizePercentage }}
            (threshold: 70%, baseline: 85%)

            CRITICAL: Model performance severely degraded

            RUNBOOK:
            1. IMMEDIATE: Increase ground truth sampling rate (validate more)
            2. Check for data drift (likely root cause)
            3. Review recent input examples (new patterns?)
            4. ESCALATE: Notify ML team for emergency retraining
            5. Consider rollback to previous model version

      # INFO: Increased low confidence predictions
      - alert: HighLowConfidencePredictions
        expr: |
          (
            rate(ml_low_confidence_predictions[1h])
            /
            rate(ml_requests_total[1h])
          ) > 0.2
        for: 1h
        labels:
          severity: info
          model: "{{ $labels.model_name }}"
        annotations:
          summary: "High rate of low confidence predictions"
          description: |
            Model {{ $labels.model_name }} low confidence rate: {{ $value | humanizePercentage }}
            (threshold: 20%)

            Model is uncertain about many predictions

            RUNBOOK:
            1. Review low confidence examples (what's different?)
            2. Check if correlated with drift
            3. Consider increasing confidence threshold (trade recall for precision)
            4. Monitor accuracy on low confidence predictions
            5. May indicate need for retraining or model improvement
```

### Alert Grouping (Reduce Noise)

```yaml
# AlertManager configuration
route:
  group_by: ['model_name', 'severity']
  group_wait: 30s        # Wait 30s before sending first alert (batch correlated)
  group_interval: 5m     # Send updates every 5 minutes
  repeat_interval: 4h    # Re-send if not resolved after 4 hours

  routes:
    # CRITICAL alerts: Page immediately
    - match:
        severity: critical
      receiver: pagerduty
      continue: true  # Also send to Slack

    # WARNING alerts: Slack notification
    - match:
        severity: warning
      receiver: slack_warnings

    # INFO alerts: Log only
    - match:
        severity: info
      receiver: slack_info

receivers:
  - name: pagerduty
    pagerduty_configs:
      - service_key: <YOUR_PAGERDUTY_KEY>
        description: "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"

  - name: slack_warnings
    slack_configs:
      - api_url: <YOUR_SLACK_WEBHOOK>
        channel: '#ml-alerts-warnings'
        title: "⚠️ ML Warning Alert"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"

  - name: slack_info
    slack_configs:
      - api_url: <YOUR_SLACK_WEBHOOK>
        channel: '#ml-alerts-info'
        title: "ℹ️ ML Info Alert"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
```

---

## Section 6: SLAs and SLOs for ML Systems

### Defining Service Level Objectives (SLOs)

```yaml
Model SLOs Template:

Service: [Model Name]
Version: [Version Number]
Owner: [Team Name]

1. LATENCY
   Objective: 95% of requests complete within [X]ms
   Measurement: P95 latency from Prometheus histogram
   Target: 95% compliance (monthly)
   Current: [Track in dashboard]

   Example:
   - P50 < 100ms
   - P95 < 500ms
   - P99 < 1000ms

2. AVAILABILITY
   Objective: Service uptime > [X]%
   Measurement: Health check success rate
   Target: 99.5% uptime (monthly) = 3.6 hours downtime allowed
   Current: [Track in dashboard]

3. ERROR RATE
   Objective: < [X]% of requests fail
   Measurement: (errors / total requests) × 100
   Target: < 1% error rate
   Current: [Track in dashboard]

4. MODEL ACCURACY
   Objective: Accuracy > [X]% on ground truth sample
   Measurement: Human-labeled sample (10% of traffic)
   Target: > 85% accuracy (rolling 1000 samples)
   Current: [Track in dashboard]

5. THROUGHPUT
   Objective: Support [X] requests/second
   Measurement: Request rate from Prometheus
   Target: Handle 1000 req/s without degradation
   Current: [Track in dashboard]

6. COST
   Objective: < $[X] per 1000 requests
   Measurement: Cloud billing / request count
   Target: < $0.05 per 1000 requests
   Current: [Track in dashboard]
```

### SLO Compliance Dashboard

```python
from prometheus_client import Gauge

SLO_COMPLIANCE_GAUGE = Gauge(
    'ml_slo_compliance_percentage',
    'SLO compliance percentage',
    ['model_name', 'slo_type']
)

class SLOTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.slos = {
            "latency_p95": {"target": 0.5, "threshold": 0.95},  # 500ms, 95% compliance
            "error_rate": {"target": 0.01, "threshold": 0.95},  # 1% errors
            "accuracy": {"target": 0.85, "threshold": 0.95},    # 85% accuracy
            "availability": {"target": 0.995, "threshold": 1.0}  # 99.5% uptime
        }
        self.measurements = {slo: [] for slo in self.slos.keys()}

    def record_measurement(self, slo_type: str, value: float):
        """Record SLO measurement (e.g., latency, error rate)"""
        self.measurements[slo_type].append({
            "value": value,
            "timestamp": time.time(),
            "compliant": self._is_compliant(slo_type, value)
        })

        # Keep last 30 days
        cutoff = time.time() - (30 * 24 * 3600)
        self.measurements[slo_type] = [
            m for m in self.measurements[slo_type]
            if m["timestamp"] > cutoff
        ]

        # Update compliance gauge
        compliance = self.calculate_compliance(slo_type)
        SLO_COMPLIANCE_GAUGE.labels(
            model_name=self.model_name,
            slo_type=slo_type
        ).set(compliance)

    def _is_compliant(self, slo_type: str, value: float) -> bool:
        """Check if single measurement meets SLO"""
        target = self.slos[slo_type]["target"]

        if slo_type in ["latency_p95", "error_rate"]:
            return value <= target  # Lower is better
        else:  # accuracy, availability
            return value >= target  # Higher is better

    def calculate_compliance(self, slo_type: str) -> float:
        """Calculate SLO compliance percentage"""
        if not self.measurements[slo_type]:
            return 0.0

        compliant_count = sum(
            1 for m in self.measurements[slo_type]
            if m["compliant"]
        )

        return compliant_count / len(self.measurements[slo_type])

    def check_slo_status(self) -> dict:
        """Check all SLOs and return status"""
        status = {}

        for slo_type, slo_config in self.slos.items():
            compliance = self.calculate_compliance(slo_type)
            threshold = slo_config["threshold"]

            status[slo_type] = {
                "compliance": compliance,
                "threshold": threshold,
                "status": "✓ MEETING SLO" if compliance >= threshold else "✗ VIOLATING SLO"
            }

        return status

# Usage
slo_tracker = SLOTracker("sentiment_classifier")

# Record measurements periodically
slo_tracker.record_measurement("latency_p95", 0.45)  # 450ms (compliant)
slo_tracker.record_measurement("error_rate", 0.008)  # 0.8% (compliant)
slo_tracker.record_measurement("accuracy", 0.87)     # 87% (compliant)

# Check overall status
status = slo_tracker.check_slo_status()
```

---

## Section 7: Monitoring Stack (Prometheus + Grafana)

### Complete Setup Example

```yaml
# docker-compose.yml

version: '3'

services:
  # ML Service
  ml-service:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"  # Metrics endpoint
    environment:
      - MODEL_PATH=/models/sentiment_classifier.pt
    volumes:
      - ./models:/models

  # Prometheus (metrics collection)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus_rules.yml:/etc/prometheus/rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana (visualization)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana_dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  # AlertManager (alert routing)
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'ml-service'
    static_configs:
      - targets: ['ml-service:8001']  # Metrics endpoint
    metrics_path: /metrics
```

```python
# ML Service with Prometheus metrics

from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Histogram
import uvicorn

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('ml_latency_seconds', 'Request latency', ['endpoint'])

@app.post("/predict")
@REQUEST_LATENCY.labels(endpoint="/predict").time()
def predict(text: str):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    result = model.predict(text)
    return {"prediction": result}

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    # Main service on port 8000
    # Metrics on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Section 8: Complete Example (End-to-End)

```python
# complete_monitoring.py
# Complete production monitoring for sentiment classifier

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from scipy.stats import ks_2samp
import numpy as np
import time
from typing import Optional

app = FastAPI()

# === 1. PERFORMANCE METRICS (RED) ===

REQUEST_COUNT = Counter(
    'sentiment_requests_total',
    'Total sentiment analysis requests',
    ['endpoint', 'model_version']
)

ERROR_COUNT = Counter(
    'sentiment_errors_total',
    'Total errors',
    ['error_type']
)

REQUEST_LATENCY = Histogram(
    'sentiment_latency_seconds',
    'Request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# === 2. MODEL QUALITY METRICS ===

PREDICTION_COUNT = Counter(
    'sentiment_predictions_by_class',
    'Predictions by sentiment class',
    ['predicted_class']
)

CONFIDENCE_HISTOGRAM = Histogram(
    'sentiment_confidence',
    'Prediction confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ACCURACY_GAUGE = Gauge(
    'sentiment_accuracy_ground_truth',
    'Accuracy on ground truth sample'
)

# === 3. DRIFT DETECTION ===

DRIFT_SCORE_GAUGE = Gauge(
    'sentiment_data_drift_ks',
    'KS statistic for data drift',
    ['feature']
)

PSI_GAUGE = Gauge(
    'sentiment_concept_drift_psi',
    'PSI for concept drift'
)

# === Initialize Monitoring Components ===

class SentimentMonitor:
    def __init__(self):
        # Reference data (from training)
        self.reference_text_lengths = np.random.normal(100, 30, 10000)

        # Drift detection
        self.current_text_lengths = []
        self.current_predictions = []
        self.baseline_prediction_dist = None

        # Ground truth tracking
        self.predictions = {}
        self.ground_truths = []

        # SLO tracking
        self.slo_measurements = []

    def track_request(self, text: str, prediction: dict, latency: float):
        """Track all metrics for a request"""
        # 1. Performance metrics
        REQUEST_COUNT.labels(
            endpoint="/predict",
            model_version="v1.0"
        ).inc()

        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        # 2. Model quality
        PREDICTION_COUNT.labels(
            predicted_class=prediction["label"]
        ).inc()

        CONFIDENCE_HISTOGRAM.observe(prediction["confidence"])

        # 3. Drift detection
        self.current_text_lengths.append(len(text))
        self.current_predictions.append(prediction["confidence"])

        # Check drift every 1000 samples
        if len(self.current_text_lengths) >= 1000:
            self.check_data_drift()
            self.check_concept_drift()
            self.current_text_lengths = []
            self.current_predictions = []

        # 4. SLO tracking
        self.slo_measurements.append({
            "latency": latency,
            "timestamp": time.time()
        })

monitor = SentimentMonitor()

# === Endpoints ===

@app.post("/predict")
def predict(text: str):
    start_time = time.time()

    try:
        # Dummy model prediction
        result = {
            "label": "positive",
            "confidence": 0.92
        }

        latency = time.time() - start_time

        # Track metrics
        monitor.track_request(text, result, latency)

        return {
            "prediction": result["label"],
            "confidence": result["confidence"],
            "latency_ms": latency * 1000
        }

    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(request_id: str, true_label: str):
    """Collect ground truth labels"""
    monitor.ground_truths.append({
        "request_id": request_id,
        "true_label": true_label,
        "timestamp": time.time()
    })

    # Calculate accuracy on last 100 samples
    if len(monitor.ground_truths) >= 100:
        recent = monitor.ground_truths[-100:]
        # Calculate accuracy (simplified)
        accuracy = 0.87  # Placeholder
        ACCURACY_GAUGE.set(accuracy)

    return {"status": "recorded"}

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Key Takeaways

1. **Monitoring is mandatory** - Instrument before deployment
2. **RED metrics first** - Rate, Errors, Duration for every service
3. **Model quality matters** - Track predictions, confidence, accuracy
4. **Drift detection prevents degradation** - KS test + PSI
5. **Actionable alerts only** - Severity-based, with runbooks
6. **SLOs define success** - Quantitative targets guide optimization
7. **Dashboard = single pane of glass** - Healthy or not in 5 seconds

**This skill prevents all 5 RED failures by providing systematic monitoring, alerting, and observability for production ML systems.**
