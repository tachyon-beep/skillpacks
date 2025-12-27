---
description: Deploy ML model to production with proper serving patterns, rollout strategy, and monitoring
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Write", "AskUserQuestion"]
argument-hint: "[model_path_or_name]"
---

# Deploy Model Command

Deploy an ML model to production with proper serving infrastructure, rollout strategy, and monitoring.

## Core Principle

**Model deployment is not "copy model file to server." It's serving infrastructure, gradual rollout, and observability from day one.**

## Information Gathering

Before deploying, determine:

1. **Serving target**: Cloud (Kubernetes), serverless, edge device?
2. **Traffic pattern**: Real-time API, batch processing, streaming?
3. **Scale requirements**: Requests/second, latency SLA?
4. **Model characteristics**: Size, inference time, dependencies?
5. **Rollout strategy**: All-at-once, canary, shadow mode?

## Deployment Decision Tree

```
Is latency critical (< 100ms)?
├─ YES → Real-time serving (REST/gRPC)
│   ├─ High throughput? → Batched inference
│   └─ Edge deployment? → Quantize first
└─ NO → Batch processing acceptable
    ├─ Scheduled jobs → Batch inference
    └─ Event-driven → Message queue + workers
```

## Serving Patterns

### Pattern 1: REST API (FastAPI + ONNX)

```python
# serve.py
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("model.onnx")

@app.post("/predict")
async def predict(data: PredictRequest):
    input_data = preprocess(data)
    outputs = session.run(None, {"input": input_data})
    return {"prediction": postprocess(outputs)}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}
```

### Pattern 2: gRPC (TorchServe)

```bash
# Package model
torch-model-archiver --model-name mymodel \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pt \
  --handler handler.py

# Start server
torchserve --start --model-store model_store --models mymodel=mymodel.mar
```

### Pattern 3: Batch Processing

```python
# batch_inference.py
from prefect import flow, task

@task
def load_batch(batch_id: str) -> pd.DataFrame:
    return load_from_storage(batch_id)

@task
def run_inference(data: pd.DataFrame, model) -> pd.DataFrame:
    return model.predict(data)

@task
def save_results(results: pd.DataFrame, batch_id: str):
    save_to_storage(results, f"predictions/{batch_id}")

@flow
def batch_inference_pipeline(batch_id: str):
    data = load_batch(batch_id)
    model = load_model()
    results = run_inference(data, model)
    save_results(results, batch_id)
```

## Containerization

### Dockerfile for ML Model

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ /app/model/
COPY serve.py /app/

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose for local testing

```yaml
version: '3.8'
services:
  model-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model
      - LOG_LEVEL=INFO
    volumes:
      - ./model:/app/model:ro
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Rollout Strategies

### Canary Deployment

```yaml
# kubernetes/canary.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"  # 10% traffic
spec:
  rules:
    - host: model.example.com
      http:
        paths:
          - path: /predict
            backend:
              service:
                name: model-v2  # New version
```

### Shadow Mode (Compare Without Risk)

```python
# shadow_deployment.py
async def predict_with_shadow(request):
    # Production prediction (returned to user)
    prod_result = await production_model.predict(request)

    # Shadow prediction (logged, not returned)
    asyncio.create_task(shadow_predict_and_log(request, prod_result))

    return prod_result

async def shadow_predict_and_log(request, prod_result):
    shadow_result = await shadow_model.predict(request)
    log_comparison(prod_result, shadow_result)
```

## Monitoring Setup

### Essential Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency')
MODEL_VERSION = Gauge('model_version_info', 'Model version', ['version'])

# Model-specific metrics
PREDICTION_DISTRIBUTION = Histogram('prediction_value', 'Prediction distribution')
INPUT_DRIFT = Gauge('input_feature_drift', 'Input feature drift score', ['feature'])
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "last_prediction_time": last_prediction_timestamp,
        "uptime_seconds": time.time() - start_time
    }
```

## Output Format

```markdown
## Model Deployment: [Model Name]

### Configuration

| Setting | Value |
|---------|-------|
| Serving Pattern | [REST/gRPC/Batch] |
| Target | [Kubernetes/Serverless/Edge] |
| Rollout Strategy | [Canary/Shadow/Direct] |
| Model Format | [ONNX/TorchServe/SavedModel] |

### Infrastructure

```yaml
[Container/Kubernetes configuration]
```

### Serving Endpoint

```python
[Serving code]
```

### Monitoring

**Metrics exposed:**
- Request latency (p50, p95, p99)
- Request count by status
- Model version
- Prediction distribution

**Health endpoint:** `/health`

### Rollout Plan

1. [ ] Deploy to staging
2. [ ] Run smoke tests
3. [ ] Deploy canary (10% traffic)
4. [ ] Monitor for 1 hour
5. [ ] Increase to 50%
6. [ ] Monitor for 1 hour
7. [ ] Full rollout (100%)

### Rollback Procedure

1. Revert Kubernetes deployment
2. Verify old version serving
3. Investigate failure

### Next Steps

- [ ] Configure autoscaling
- [ ] Set up alerting
- [ ] Add A/B testing capability
```

## Cross-Pack Discovery

```python
import glob

# For deployment strategies
devops_pack = glob.glob("plugins/axiom-devops-engineering/plugin.json")
if devops_pack:
    print("Available: axiom-devops-engineering for deployment strategies")

# For monitoring
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("Available: ordis-quality-engineering for observability patterns")
```

## Load Detailed Guidance

For serving patterns:
```
Load skill: yzmir-ml-production:using-ml-production
Then read: model-serving-patterns.md
```

For deployment strategies:
```
Then read: deployment-strategies.md
```
