---
description: Diagnose production ML issues - slow inference, accuracy degradation, drift, and errors
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[symptom_or_model_name]"
---

# Diagnose Inference Command

Diagnose production ML issues including slow inference, accuracy degradation, drift, and errors.

## Core Principle

**Production ML failures are usually not model bugs. Check infrastructure, data, and monitoring before blaming the model.**

## Common Symptoms and Causes

| Symptom | Likely Causes | First Check |
|---------|---------------|-------------|
| **Slow inference** | Batch size, hardware, model size | Profile request |
| **Accuracy drop** | Data drift, concept drift, stale model | Compare distributions |
| **High latency variance** | Cold starts, GC, resource contention | Check p99 vs p50 |
| **OOM errors** | Batch size, memory leaks, model size | Monitor memory usage |
| **Intermittent failures** | Resource exhaustion, race conditions | Check concurrent load |
| **Wrong predictions** | Input preprocessing, model version | Validate input pipeline |

## Diagnostic Protocol

### Step 1: Reproduce and Isolate

```bash
# Check if issue is model or infrastructure
# Test with known-good input
curl -X POST http://model-server:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"known_input": "expected_output"}'

# Check response time
curl -w "Time: %{time_total}s\n" -o /dev/null -s http://model-server:8000/predict

# Check under load
ab -n 100 -c 10 http://model-server:8000/predict
```

### Step 2: Check Infrastructure

```bash
# Memory usage
kubectl top pods -l app=model-server

# GPU utilization (if applicable)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Container logs
kubectl logs -l app=model-server --tail=100 | grep -E "ERROR|WARN|OOM"

# Resource limits
kubectl describe pod model-server | grep -A5 "Limits:"
```

### Step 3: Profile Inference

```python
# Add profiling to identify bottleneck
import time

def profiled_predict(input_data):
    timings = {}

    start = time.perf_counter()
    preprocessed = preprocess(input_data)
    timings['preprocess'] = time.perf_counter() - start

    start = time.perf_counter()
    model_output = model.predict(preprocessed)
    timings['model_inference'] = time.perf_counter() - start

    start = time.perf_counter()
    result = postprocess(model_output)
    timings['postprocess'] = time.perf_counter() - start

    return result, timings
```

**Common bottlenecks:**
| Component | Typical Time | If Slow |
|-----------|--------------|---------|
| Preprocessing | < 10ms | Optimize transforms |
| Model inference | 10-100ms | Quantize, batch, GPU |
| Postprocessing | < 10ms | Simplify output |
| Network/IO | < 5ms | Check serialization |

### Step 4: Check Data Drift

```python
# Compare production input distribution to training distribution
from scipy import stats

def check_drift(production_data, reference_data, threshold=0.05):
    drift_scores = {}
    for feature in production_data.columns:
        statistic, p_value = stats.ks_2samp(
            production_data[feature],
            reference_data[feature]
        )
        drift_scores[feature] = {
            'statistic': statistic,
            'p_value': p_value,
            'drifted': p_value < threshold
        }
    return drift_scores
```

**Drift types:**
| Type | Symptom | Detection | Fix |
|------|---------|-----------|-----|
| **Data drift** | Input distribution changed | KS test, PSI | Retrain on new data |
| **Concept drift** | Relationship changed | Monitor accuracy | Retrain model |
| **Label drift** | Target distribution changed | Monitor predictions | Investigate cause |

### Step 5: Check Model Version

```bash
# Verify correct model is deployed
curl http://model-server:8000/health | jq '.model_version'

# Check model file timestamps
ls -la /model/*.pt /model/*.onnx

# Verify model hash
md5sum /model/model.onnx
```

## Issue-Specific Diagnostics

### Slow Inference

```python
# Profile with PyTorch
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True
) as prof:
    model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Common fixes:**
- Enable batching
- Use ONNX Runtime / TensorRT
- Quantize model (INT8)
- Increase GPU memory
- Enable CUDA graphs

### OOM Errors

```bash
# Check memory growth
watch -n 1 'nvidia-smi | grep MiB'

# Check for memory leaks in Python
tracemalloc.start()
# ... run predictions ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')[:10]
```

**Common fixes:**
- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use gradient checkpointing
- Quantize to smaller precision

### Accuracy Degradation

```python
# Compare prediction distributions
import numpy as np

def compare_predictions(current, historical):
    # Check if distribution shifted
    current_mean = np.mean(current)
    historical_mean = np.mean(historical)

    shift = abs(current_mean - historical_mean) / historical_mean
    print(f"Prediction mean shift: {shift:.2%}")

    if shift > 0.10:
        print("WARNING: Significant prediction shift detected")
```

**Investigation checklist:**
1. [ ] Compare input distribution to training data
2. [ ] Check preprocessing pipeline unchanged
3. [ ] Verify model version is correct
4. [ ] Check for upstream data changes
5. [ ] Compare with shadow model if available

## Output Format

```markdown
## Inference Diagnostic Report: [Model Name]

### Issue Summary

**Symptom**: [What's happening]
**Severity**: [Critical/High/Medium]
**Duration**: [When started, how long]

### Root Cause Analysis

| Hypothesis | Evidence | Confirmed? |
|------------|----------|------------|
| [Cause 1] | [Data] | Yes/No |
| [Cause 2] | [Data] | Yes/No |

### Findings

**Infrastructure:**
- CPU/GPU utilization: [Value]
- Memory usage: [Value]
- Latency p50/p95/p99: [Values]

**Model:**
- Version: [Version]
- Inference time: [Value]
- Prediction distribution: [Normal/Shifted]

**Data:**
- Input drift detected: [Yes/No, which features]
- Preprocessing: [Correct/Issue]

### Root Cause

**Component**: [Infrastructure/Model/Data/Pipeline]
**Cause**: [Specific issue]
**Evidence**: [Supporting data]

### Solution

**Immediate fix:**
[Action to take]

**Long-term fix:**
[Preventive measures]

### Prevention

- [ ] Add monitoring for [metric]
- [ ] Set alert threshold for [condition]
- [ ] Add test case for [scenario]
```

## Cross-Pack Discovery

```python
import glob

# For PyTorch profiling
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/plugin.json")
if pytorch_pack:
    print("For PyTorch profiling: use yzmir-pytorch-engineering")

# For monitoring setup
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("For observability patterns: use ordis-quality-engineering")
```

## Load Detailed Guidance

For debugging techniques:
```
Load skill: yzmir-ml-production:using-ml-production
Then read: production-debugging-techniques.md
```

For monitoring setup:
```
Then read: production-monitoring-and-alerting.md
```
