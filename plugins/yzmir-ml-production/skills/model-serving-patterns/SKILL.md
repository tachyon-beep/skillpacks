---
name: model-serving-patterns
description: Design production model serving infrastructure with FastAPI, TorchServe, gRPC, ONNX Runtime, request batching, and containerization. Use when deploying ML models to production with performance and reliability requirements.
---

# Model Serving Patterns Skill

## When to Use This Skill

Use this skill when:
- Deploying ML models to production environments
- Building model serving APIs for real-time inference
- Optimizing model serving for throughput and latency
- Containerizing models for consistent deployment
- Implementing request batching for efficiency
- Choosing between serving frameworks and protocols

**When NOT to use:** Notebook prototyping, training jobs, or single-prediction scripts where serving infrastructure is premature.

## Core Principle

**Serving infrastructure is not one-size-fits-all. Pattern selection is context-dependent.**

Without proper serving infrastructure:
- model.pkl in repo (manual dependency hell)
- Wrong protocol choice (gRPC for simple REST use cases)
- No batching (1 req/sec instead of 100 req/sec)
- Not containerized (works on my machine syndrome)
- Static batching when dynamic needed (underutilized GPU)

**Formula:** Right framework (FastAPI vs TorchServe vs gRPC vs ONNX) + Request batching (dynamic > static) + Containerization (Docker + model) + Clear selection criteria = Production-ready serving.

## Serving Framework Decision Tree

```
┌────────────────────────────────────────┐
│     What's your primary requirement?   │
└──────────────┬─────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
   Flexibility    Batteries Included
       │               │
       ▼               ▼
   FastAPI         TorchServe
   (Custom)        (PyTorch)
       │               │
       │       ┌───────┴───────┐
       │       ▼               ▼
       │   Low Latency   Cross-Framework
       │       │               │
       │       ▼               ▼
       │    gRPC         ONNX Runtime
       │       │               │
       └───────┴───────────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │  Add Request Batching  │
       │  Dynamic > Static      │
       └───────────┬────────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   Containerize with    │
       │   Docker + Dependencies│
       └────────────────────────┘
```

## Part 1: FastAPI for Custom Serving

**When to use:** Need flexibility, custom preprocessing, or non-standard workflows.

**Advantages:** Full control, easy debugging, Python ecosystem integration.
**Disadvantages:** Manual optimization, no built-in model management.

### Basic FastAPI Serving

```python
# serve_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Serving API", version="1.0.0")

class PredictionRequest(BaseModel):
    """Request schema with validation."""
    inputs: List[List[float]] = Field(..., description="Input features as 2D array")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    class Config:
        schema_extra = {
            "example": {
                "inputs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "return_probabilities": True
            }
        }

class PredictionResponse(BaseModel):
    """Response schema."""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    latency_ms: float

class ModelServer:
    """
    Model server with lazy loading and caching.

    WHY: Load model once at startup, reuse across requests.
    WHY: Avoids 5-10 second model loading per request.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load model on first request (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()  # WHY: Disable dropout, batchnorm for inference
            logger.info("Model loaded successfully")

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference.

        Args:
            inputs: Input array (batch_size, features)

        Returns:
            (predictions, probabilities)
        """
        self.load_model()

        # Convert to tensor
        x = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        # WHY: torch.no_grad() disables gradient computation for inference
        # WHY: Reduces memory usage by 50% and speeds up by 2×
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

# Global model server instance
model_server = ModelServer(model_path="model.pth")

@app.on_event("startup")
async def startup_event():
    """Load model at startup for faster first request."""
    model_server.load_model()
    logger.info("Server startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None,
        "device": str(model_server.device)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint with validation and error handling.

    WHY: Pydantic validates inputs automatically.
    WHY: Returns 422 for invalid inputs, not 500.
    """
    import time
    start_time = time.time()

    try:
        inputs = np.array(request.inputs)

        # Validate shape
        if inputs.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D array, got {inputs.ndim}D"
            )

        predictions, probabilities = model_server.predict(inputs)

        latency_ms = (time.time() - start_time) * 1000

        response = PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist() if request.return_probabilities else None,
            latency_ms=latency_ms
        )

        logger.info(f"Predicted {len(predictions)} samples in {latency_ms:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000 --workers 4
```

**Performance characteristics:**

| Metric | Value | Notes |
|--------|-------|-------|
| Cold start | 5-10s | Model loading time |
| Warm latency | 10-50ms | Per request |
| Throughput | 100-500 req/sec | Single worker |
| Memory | 2-8GB | Model + runtime |

### Advanced: Async FastAPI with Background Tasks

```python
# serve_fastapi_async.py
from fastapi import FastAPI, BackgroundTasks
from asyncio import Queue, create_task, sleep
import asyncio
from typing import Dict
import uuid

app = FastAPI()

class AsyncBatchPredictor:
    """
    Async batch predictor with request queuing.

    WHY: Collect multiple requests, predict as batch.
    WHY: GPU utilization: 20% (1 req) → 80% (batch of 32).
    """

    def __init__(self, model_server: ModelServer, batch_size: int = 32, wait_ms: int = 10):
        self.model_server = model_server
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.queue: Queue = Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def start(self):
        """Start background batch processing loop."""
        create_task(self._batch_processing_loop())

    async def _batch_processing_loop(self):
        """
        Continuously collect and process batches.

        WHY: Wait for batch_size OR timeout, then process.
        WHY: Balances throughput (large batch) and latency (timeout).
        """
        while True:
            batch_requests = []
            batch_ids = []

            # Collect batch
            deadline = asyncio.get_event_loop().time() + (self.wait_ms / 1000)

            while len(batch_requests) < self.batch_size:
                timeout = max(0, deadline - asyncio.get_event_loop().time())

                try:
                    request_id, inputs = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=timeout
                    )
                    batch_requests.append(inputs)
                    batch_ids.append(request_id)
                except asyncio.TimeoutError:
                    break  # Timeout reached, process what we have

            if not batch_requests:
                await sleep(0.001)  # Brief sleep before next iteration
                continue

            # Process batch
            batch_array = np.array(batch_requests)
            predictions, probabilities = self.model_server.predict(batch_array)

            # Return results to waiting requests
            for i, request_id in enumerate(batch_ids):
                future = self.pending_requests.pop(request_id)
                future.set_result((predictions[i], probabilities[i]))

    async def predict_async(self, inputs: List[float]) -> tuple[int, np.ndarray]:
        """
        Add request to queue and await result.

        WHY: Returns immediately if batch ready, waits if not.
        WHY: Client doesn't know about batching (transparent).
        """
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        await self.queue.put((request_id, inputs))

        # Wait for batch processing to complete
        prediction, probability = await future
        return prediction, probability

# Global async predictor
async_predictor = None

@app.on_event("startup")
async def startup():
    global async_predictor
    model_server.load_model()
    async_predictor = AsyncBatchPredictor(model_server, batch_size=32, wait_ms=10)
    await async_predictor.start()

@app.post("/predict_async")
async def predict_async(request: PredictionRequest):
    """
    Async prediction with automatic batching.

    WHY: 10× better GPU utilization than synchronous.
    WHY: Same latency, much higher throughput.
    """
    # Single input for simplicity (extend for batch)
    inputs = request.inputs[0]
    prediction, probability = await async_predictor.predict_async(inputs)

    return {
        "prediction": int(prediction),
        "probability": probability.tolist()
    }
```

**Performance improvement:**

| Approach | Throughput | GPU Utilization | Latency P95 |
|----------|-----------|-----------------|-------------|
| Synchronous | 100 req/sec | 20% | 15ms |
| Async batching | 1000 req/sec | 80% | 25ms |
| Improvement | **10×** | **4×** | +10ms |

---

## Part 2: TorchServe for PyTorch Models

**When to use:** PyTorch models, want batteries-included solution with monitoring, metrics, and model management.

**Advantages:** Built-in batching, model versioning, A/B testing, metrics.
**Disadvantages:** PyTorch-only, less flexibility, steeper learning curve.

### Creating a TorchServe Handler

```python
# handler.py
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import logging

logger = logging.getLogger(__name__)

class CustomClassifierHandler(BaseHandler):
    """
    Custom TorchServe handler with preprocessing and batching.

    WHY: TorchServe provides: model versioning, A/B testing, metrics, monitoring.
    WHY: Built-in dynamic batching (no custom code needed).
    """

    def initialize(self, context):
        """
        Initialize handler (called once at startup).

        Args:
            context: TorchServe context with model artifacts
        """
        self.manifest = context.manifest
        properties = context.system_properties

        # Set device
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load model
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = f"{model_dir}/{serialized_file}"

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

        # WHY: Initialize preprocessing parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess input data.

        Args:
            data: List of input requests

        Returns:
            Preprocessed tensor batch

        WHY: TorchServe batches requests automatically.
        WHY: This method receives multiple requests at once.
        """
        inputs = []

        for row in data:
            # Get input from request (JSON or binary)
            input_data = row.get("data") or row.get("body")

            # Parse and convert
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")

            # Convert to tensor
            tensor = torch.tensor(eval(input_data), dtype=torch.float32)

            # Normalize
            tensor = (tensor - self.mean) / self.std

            inputs.append(tensor)

        # Stack into batch
        batch = torch.stack(inputs).to(self.device)
        return batch

    def inference(self, batch):
        """
        Run inference on batch.

        Args:
            batch: Preprocessed batch tensor

        Returns:
            Model output

        WHY: torch.no_grad() for inference (faster, less memory).
        """
        with torch.no_grad():
            output = self.model(batch)

        return output

    def postprocess(self, inference_output):
        """
        Postprocess inference output.

        Args:
            inference_output: Raw model output

        Returns:
            List of predictions (one per request in batch)

        WHY: Convert tensors to JSON-serializable format.
        WHY: Return predictions in same order as inputs.
        """
        # Apply softmax
        probabilities = F.softmax(inference_output, dim=1)

        # Get predictions
        predictions = torch.argmax(probabilities, dim=1)

        # Convert to list (one entry per request)
        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": predictions[i].item(),
                "probabilities": probabilities[i].tolist()
            })

        return results
```

### TorchServe Configuration

```python
# model_config.yaml
# WHY: Configuration controls batching, workers, timeouts
# WHY: Tune these for your latency/throughput requirements

minWorkers: 2          # WHY: Minimum workers (always ready)
maxWorkers: 4          # WHY: Maximum workers (scale up under load)
batchSize: 32          # WHY: Maximum batch size (GPU utilization)
maxBatchDelay: 10      # WHY: Max wait time for batch (ms)
                       # WHY: Trade-off: larger batch (better GPU util) vs latency

responseTimeout: 120   # WHY: Request timeout (seconds)
                       # WHY: Prevent hung requests

# Device assignment
deviceType: "gpu"      # WHY: Use GPU if available
deviceIds: [0]         # WHY: Specific GPU ID

# Metrics
metrics:
  enable: true
  prometheus: true     # WHY: Export to Prometheus for monitoring
```

### Packaging and Serving

```bash
# Package model for TorchServe
# WHY: .mar file contains model + handler + config (portable)
torch-model-archiver \
  --model-name classifier \
  --version 1.0 \
  --serialized-file model.pt \
  --handler handler.py \
  --extra-files "model_config.yaml" \
  --export-path model_store/

# Start TorchServe
# WHY: Serves on 8080 (inference), 8081 (management), 8082 (metrics)
torchserve \
  --start \
  --ncs \
  --model-store model_store \
  --models classifier.mar \
  --ts-config config.properties

# Register model (if not auto-loaded)
curl -X POST "http://localhost:8081/models?url=classifier.mar&batch_size=32&max_batch_delay=10"

# Make prediction
curl -X POST "http://localhost:8080/predictions/classifier" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0]]}'

# Get metrics (for monitoring)
curl http://localhost:8082/metrics

# Unregister model (for updates)
curl -X DELETE "http://localhost:8081/models/classifier"
```

**TorchServe advantages:**

| Feature | Built-in? | Notes |
|---------|-----------|-------|
| Dynamic batching | ✓ | Automatic, configurable |
| Model versioning | ✓ | A/B testing support |
| Metrics/monitoring | ✓ | Prometheus integration |
| Multi-model serving | ✓ | Multiple models per server |
| GPU management | ✓ | Automatic device assignment |
| Custom preprocessing | ✓ | Via handler |

---

## Part 3: gRPC for Low-Latency Serving

**When to use:** Low latency critical (< 10ms), internal services, microservices architecture.

**Advantages:** 3-5× faster than REST, binary protocol, streaming support.
**Disadvantages:** More complex, requires proto definitions, harder debugging.

### Protocol Definition

```protobuf
// model_service.proto
syntax = "proto3";

package modelserving;

// WHY: Define service contract in .proto file
// WHY: Code generation for multiple languages (Python, Go, Java, etc.)
service ModelService {
  // Unary RPC (one request, one response)
  rpc Predict (PredictRequest) returns (PredictResponse);

  // Server streaming (one request, stream responses)
  rpc PredictStream (PredictRequest) returns (stream PredictResponse);

  // Bidirectional streaming (stream requests and responses)
  rpc PredictBidi (stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  // WHY: Repeated = array/list
  repeated float features = 1;  // WHY: Input features
  bool return_probabilities = 2;
}

message PredictResponse {
  int32 prediction = 1;
  repeated float probabilities = 2;
  float latency_ms = 3;
}

// Health check service (for load balancers)
service Health {
  rpc Check (HealthCheckRequest) returns (HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
  }
  ServingStatus status = 1;
}
```

### gRPC Server Implementation

```python
# serve_grpc.py
import grpc
from concurrent import futures
import time
import logging
import torch
import numpy as np

# Generated from proto file (run: python -m grpc_tools.protoc ...)
import model_service_pb2
import model_service_pb2_grpc

logger = logging.getLogger(__name__)

class ModelServicer(model_service_pb2_grpc.ModelServiceServicer):
    """
    gRPC service implementation.

    WHY: gRPC is 3-5× faster than REST (binary protocol, HTTP/2).
    WHY: Use for low-latency internal services (< 10ms target).
    """

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def Predict(self, request, context):
        """
        Unary RPC prediction.

        WHY: Fastest for single predictions.
        WHY: 3-5ms latency vs 10-15ms for REST.
        """
        start_time = time.time()

        try:
            # Convert proto repeated field to numpy
            features = np.array(request.features, dtype=np.float32)

            # Reshape for model
            x = torch.tensor(features).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

            latency_ms = (time.time() - start_time) * 1000

            # Build response
            response = model_service_pb2.PredictResponse(
                prediction=int(pred.item()),
                latency_ms=latency_ms
            )

            # WHY: Only include probabilities if requested (reduce bandwidth)
            if request.return_probabilities:
                response.probabilities.extend(probs[0].cpu().tolist())

            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.PredictResponse()

    def PredictStream(self, request, context):
        """
        Server streaming RPC.

        WHY: Send multiple predictions over one connection.
        WHY: Lower overhead for batch processing.
        """
        # Stream multiple predictions (example: time series)
        for i in range(10):  # Simulate 10 predictions
            response = self.Predict(request, context)
            yield response
            time.sleep(0.01)  # Simulate processing delay

    def PredictBidi(self, request_iterator, context):
        """
        Bidirectional streaming RPC.

        WHY: Real-time inference (send request, get response immediately).
        WHY: Lowest latency for streaming use cases.
        """
        for request in request_iterator:
            response = self.Predict(request, context)
            yield response

class HealthServicer(model_service_pb2_grpc.HealthServicer):
    """Health check service for load balancers."""

    def Check(self, request, context):
        # WHY: Load balancers need health checks to route traffic
        return model_service_pb2.HealthCheckResponse(
            status=model_service_pb2.HealthCheckResponse.SERVING
        )

def serve():
    """
    Start gRPC server.

    WHY: ThreadPoolExecutor for concurrent request handling.
    WHY: max_workers controls concurrency (tune based on CPU cores).
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            # WHY: These options optimize for low latency
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ('grpc.so_reuseport', 1),  # WHY: Allows multiple servers on same port
            ('grpc.use_local_subchannel_pool', 1)  # WHY: Better connection reuse
        ]
    )

    # Add services
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServicer("model.pth"), server
    )
    model_service_pb2_grpc.add_HealthServicer_to_server(
        HealthServicer(), server
    )

    # Bind to port
    server.add_insecure_port('[::]:50051')

    server.start()
    logger.info("gRPC server started on port 50051")

    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
```

### gRPC Client

```python
# client_grpc.py
import grpc
import model_service_pb2
import model_service_pb2_grpc
import time

def benchmark_grpc_vs_rest():
    """
    Benchmark gRPC vs REST latency.

    WHY: gRPC is faster, but how much faster?
    """
    # gRPC client
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_service_pb2_grpc.ModelServiceStub(channel)

    # Warm up
    request = model_service_pb2.PredictRequest(
        features=[1.0, 2.0, 3.0],
        return_probabilities=True
    )
    for _ in range(10):
        stub.Predict(request)

    # Benchmark
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        response = stub.Predict(request)
    grpc_latency = ((time.time() - start) / iterations) * 1000

    print(f"gRPC average latency: {grpc_latency:.2f}ms")

    # Compare with REST (FastAPI)
    import requests
    rest_url = "http://localhost:8000/predict"

    # Warm up
    for _ in range(10):
        requests.post(rest_url, json={"inputs": [[1.0, 2.0, 3.0]]})

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        requests.post(rest_url, json={"inputs": [[1.0, 2.0, 3.0]]})
    rest_latency = ((time.time() - start) / iterations) * 1000

    print(f"REST average latency: {rest_latency:.2f}ms")
    print(f"gRPC is {rest_latency/grpc_latency:.1f}× faster")

    # Typical results:
    # gRPC: 3-5ms
    # REST: 10-15ms
    # gRPC is 3-5× faster

if __name__ == "__main__":
    benchmark_grpc_vs_rest()
```

**gRPC vs REST comparison:**

| Metric | gRPC | REST | Advantage |
|--------|------|------|-----------|
| Latency | 3-5ms | 10-15ms | **gRPC 3× faster** |
| Throughput | 10k req/sec | 3k req/sec | **gRPC 3× higher** |
| Payload size | Binary (smaller) | JSON (larger) | gRPC 30-50% smaller |
| Debugging | Harder | Easier | REST |
| Browser support | No (requires proxy) | Yes | REST |
| Streaming | Native | Complex (SSE/WebSocket) | gRPC |

---

## Part 4: ONNX Runtime for Cross-Framework Serving

**When to use:** Need cross-framework support (PyTorch, TensorFlow, etc.), want maximum performance, or deploying to edge devices.

**Advantages:** Framework-agnostic, highly optimized, smaller deployment size.
**Disadvantages:** Not all models convert easily, limited debugging.

### Converting PyTorch to ONNX

```python
# convert_to_onnx.py
import torch
import torch.onnx

def convert_pytorch_to_onnx(model_path: str, output_path: str):
    """
    Convert PyTorch model to ONNX format.

    WHY: ONNX is framework-agnostic (portable).
    WHY: ONNX Runtime is 2-3× faster than native PyTorch inference.
    WHY: Smaller deployment size (no PyTorch dependency).
    """
    # Load PyTorch model
    model = torch.load(model_path)
    model.eval()

    # Create dummy input (for tracing)
    dummy_input = torch.randn(1, 3, 224, 224)  # Example: image

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,  # WHY: Include model weights
        opset_version=17,    # WHY: Latest stable ONNX opset
        do_constant_folding=True,  # WHY: Optimize constants at export time
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # WHY: Support variable batch size
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {output_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful")

# Example usage
convert_pytorch_to_onnx("model.pth", "model.onnx")
```

### ONNX Runtime Serving

```python
# serve_onnx.py
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

class ONNXModelServer:
    """
    ONNX Runtime server with optimizations.

    WHY: ONNX Runtime is 2-3× faster than PyTorch inference.
    WHY: Smaller memory footprint (no PyTorch/TensorFlow).
    WHY: Cross-platform (Windows, Linux, Mac, mobile, edge).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load_model(self):
        """Load ONNX model with optimizations."""
        if self.session is None:
            # Set execution providers (GPU > CPU)
            # WHY: Tries GPU first, falls back to CPU
            providers = [
                'CUDAExecutionProvider',  # NVIDIA GPU
                'CPUExecutionProvider'     # CPU fallback
            ]

            # Session options for optimization
            sess_options = ort.SessionOptions()

            # WHY: Enable graph optimizations (fuse ops, constant folding)
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # WHY: Intra-op parallelism (parallel ops within graph)
            sess_options.intra_op_num_threads = 4

            # WHY: Inter-op parallelism (parallel independent subgraphs)
            sess_options.inter_op_num_threads = 2

            # WHY: Enable memory pattern optimization
            sess_options.enable_mem_pattern = True

            # WHY: Enable CPU memory arena (reduces allocation overhead)
            sess_options.enable_cpu_mem_arena = True

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"ONNX model loaded: {self.model_path}")
            logger.info(f"Execution provider: {self.session.get_providers()[0]}")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference.

        WHY: ONNX Runtime automatically optimizes:
        - Operator fusion (combine multiple ops)
        - Constant folding (compute constants at load time)
        - Memory reuse (reduce allocations)
        """
        self.load_model()

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs.astype(np.float32)}
        )

        return outputs[0]

    def benchmark_vs_pytorch(self, num_iterations: int = 1000):
        """Compare ONNX vs PyTorch inference speed."""
        import time
        import torch

        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Warm up
        for _ in range(10):
            self.predict(dummy_input)

        # Benchmark ONNX
        start = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_input)
        onnx_time = (time.time() - start) / num_iterations * 1000

        # Benchmark PyTorch
        pytorch_model = torch.load(self.model_path.replace('.onnx', '.pth'))
        pytorch_model.eval()

        dummy_tensor = torch.from_numpy(dummy_input)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                pytorch_model(dummy_tensor)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                pytorch_model(dummy_tensor)
        pytorch_time = (time.time() - start) / num_iterations * 1000

        print(f"ONNX Runtime: {onnx_time:.2f}ms")
        print(f"PyTorch: {pytorch_time:.2f}ms")
        print(f"ONNX is {pytorch_time/onnx_time:.1f}× faster")

        # Typical results:
        # ONNX: 5-8ms
        # PyTorch: 12-20ms
        # ONNX is 2-3× faster

# Global server
onnx_server = ONNXModelServer("model.onnx")

@app.on_event("startup")
async def startup():
    onnx_server.load_model()

@app.post("/predict")
async def predict(request: PredictionRequest):
    """ONNX prediction endpoint."""
    inputs = np.array(request.inputs, dtype=np.float32)
    outputs = onnx_server.predict(inputs)

    return {
        "predictions": outputs.tolist()
    }
```

**ONNX Runtime advantages:**

| Feature | Benefit | Measurement |
|---------|---------|-------------|
| Speed | Optimized operators | 2-3× faster than native |
| Size | No framework dependency | 10-50MB vs 500MB+ (PyTorch) |
| Portability | Framework-agnostic | PyTorch/TF/etc → ONNX |
| Edge deployment | Lightweight runtime | Mobile, IoT, embedded |

---

## Part 5: Request Batching Patterns

**Core principle:** Batch requests for GPU efficiency.

**Why batching matters:**
- GPU utilization: 20% (single request) → 80% (batch of 32)
- Throughput: 100 req/sec (unbatched) → 1000 req/sec (batched)
- Cost: 10× reduction in GPU cost per request

### Dynamic Batching (Adaptive)

```python
# dynamic_batching.py
import asyncio
from asyncio import Queue, Lock
from typing import List, Tuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class DynamicBatcher:
    """
    Dynamic batching with adaptive timeout.

    WHY: Static batching waits for full batch (high latency at low load).
    WHY: Dynamic batching adapts: full batch OR timeout (balanced).

    Key parameters:
    - max_batch_size: Maximum batch size (GPU memory limit)
    - max_wait_ms: Maximum wait time (latency target)

    Trade-off:
    - Larger batch → better GPU utilization, higher throughput
    - Shorter timeout → lower latency, worse GPU utilization
    """

    def __init__(
        self,
        model_server,
        max_batch_size: int = 32,
        max_wait_ms: int = 10
    ):
        self.model_server = model_server
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.request_queue: Queue = Queue()
        self.batch_lock = Lock()

        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "gpu_utilization": 0
        }

    async def start(self):
        """Start batch processing loop."""
        asyncio.create_task(self._batch_loop())

    async def _batch_loop(self):
        """
        Main batching loop.

        Algorithm:
        1. Wait for first request
        2. Start timeout timer
        3. Collect requests until:
           - Batch full (max_batch_size reached)
           - OR timeout expired (max_wait_ms)
        4. Process batch
        5. Return results to waiting requests
        """
        while True:
            batch = []
            futures = []

            # Wait for first request (no timeout)
            request_data, future = await self.request_queue.get()
            batch.append(request_data)
            futures.append(future)

            # Start deadline timer
            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000)

            # Collect additional requests until batch full or timeout
            while len(batch) < self.max_batch_size:
                remaining_time = max(0, deadline - asyncio.get_event_loop().time())

                try:
                    request_data, future = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=remaining_time
                    )
                    batch.append(request_data)
                    futures.append(future)
                except asyncio.TimeoutError:
                    # Timeout: process what we have
                    break

            # Process batch
            await self._process_batch(batch, futures)

    async def _process_batch(
        self,
        batch: List[np.ndarray],
        futures: List[asyncio.Future]
    ):
        """Process batch and return results."""
        batch_size = len(batch)

        # Convert to batch array
        batch_array = np.array(batch)

        # Run inference
        start_time = time.time()
        predictions, probabilities = self.model_server.predict(batch_array)
        inference_time = (time.time() - start_time) * 1000

        # Update stats
        self.stats["total_requests"] += batch_size
        self.stats["total_batches"] += 1
        self.stats["avg_batch_size"] = (
            self.stats["total_requests"] / self.stats["total_batches"]
        )
        self.stats["gpu_utilization"] = (
            self.stats["avg_batch_size"] / self.max_batch_size * 100
        )

        logger.info(
            f"Processed batch: size={batch_size}, "
            f"inference_time={inference_time:.2f}ms, "
            f"avg_batch_size={self.stats['avg_batch_size']:.1f}, "
            f"gpu_util={self.stats['gpu_utilization']:.1f}%"
        )

        # Return results to waiting requests
        for i, future in enumerate(futures):
            if not future.done():
                future.set_result((predictions[i], probabilities[i]))

    async def predict(self, inputs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Add request to batch queue.

        WHY: Transparent batching (caller doesn't see batching).
        WHY: Returns when batch processed (might wait for other requests).
        """
        future = asyncio.Future()
        await self.request_queue.put((inputs, future))

        # Wait for batch to be processed
        prediction, probability = await future
        return prediction, probability

    def get_stats(self):
        """Get batching statistics."""
        return self.stats

# Example usage with load simulation
async def simulate_load():
    """
    Simulate varying load to demonstrate dynamic batching.

    WHY: Shows how batcher adapts to load:
    - High load: Fills batches quickly (high GPU util)
    - Low load: Processes smaller batches (low latency)
    """
    from serve_fastapi import ModelServer

    model_server = ModelServer("model.pth")
    model_server.load_model()

    batcher = DynamicBatcher(
        model_server,
        max_batch_size=32,
        max_wait_ms=10
    )
    await batcher.start()

    # High load (32 concurrent requests)
    print("Simulating HIGH LOAD (32 concurrent)...")
    tasks = []
    for i in range(32):
        inputs = np.random.randn(10)
        task = asyncio.create_task(batcher.predict(inputs))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    print(f"High load results: {len(results)} predictions")
    print(f"Stats: {batcher.get_stats()}")
    # Expected: avg_batch_size ≈ 32, gpu_util ≈ 100%

    await asyncio.sleep(0.1)  # Reset

    # Low load (1 request at a time)
    print("\nSimulating LOW LOAD (1 at a time)...")
    for i in range(10):
        inputs = np.random.randn(10)
        result = await batcher.predict(inputs)
        await asyncio.sleep(0.02)  # 20ms between requests

    print(f"Stats: {batcher.get_stats()}")
    # Expected: avg_batch_size ≈ 1-2, gpu_util ≈ 5-10%
    # WHY: Timeout expires before batch fills (low latency maintained)

if __name__ == "__main__":
    asyncio.run(simulate_load())
```

**Batching performance:**

| Load | Batch Size | GPU Util | Latency | Throughput |
|------|-----------|----------|---------|------------|
| High (100 req/sec) | 28-32 | 90% | 12ms | 1000 req/sec |
| Medium (20 req/sec) | 8-12 | 35% | 11ms | 200 req/sec |
| Low (5 req/sec) | 1-2 | 10% | 10ms | 50 req/sec |

**Key insight:** Dynamic batching adapts to load while maintaining latency target.

---

## Part 6: Containerization

**Why containerize:** "Works on my machine" → "Works everywhere"

**Benefits:**
- Reproducible builds (same dependencies, versions)
- Isolated environment (no conflicts)
- Portable deployment (dev, staging, prod identical)
- Easy scaling (K8s, Docker Swarm)

### Multi-Stage Docker Build

```dockerfile
# Dockerfile
# WHY: Multi-stage build reduces image size by 50-80%
# WHY: Build stage has compilers, runtime stage only has runtime deps

# ==================== Stage 1: Build ====================
FROM python:3.11-slim as builder

# WHY: Install build dependencies (needed for compilation)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# WHY: Create virtual environment in builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# WHY: Copy only requirements first (layer caching)
# WHY: If requirements.txt unchanged, this layer is cached
COPY requirements.txt .

# WHY: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime ====================
FROM python:3.11-slim

# WHY: Copy only virtual environment from builder (not build tools)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# WHY: Set working directory
WORKDIR /app

# WHY: Copy application code
COPY serve_fastapi.py .
COPY model.pth .

# WHY: Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# WHY: Expose port (documentation, not enforcement)
EXPOSE 8000

# WHY: Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# WHY: Run with uvicorn (production ASGI server)
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Multi-Service

```yaml
# docker-compose.yml
# WHY: Docker Compose for local development and testing
# WHY: Defines multiple services (API, model, monitoring)

version: '3.8'

services:
  # Model serving API
  model-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      # WHY: Environment variables for configuration
      - MODEL_PATH=/app/model.pth
      - LOG_LEVEL=INFO
    volumes:
      # WHY: Mount model directory (for updates without rebuild)
      - ./models:/app/models:ro
    deploy:
      resources:
        # WHY: Limit resources to prevent resource exhaustion
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          # WHY: Reserve GPU (requires nvidia-docker)
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

### Build and Deploy

```bash
# Build image
# WHY: Tag with version for rollback capability
docker build -t model-api:1.0.0 .

# Run container
docker run -d \
  --name model-api \
  -p 8000:8000 \
  --gpus all \
  model-api:1.0.0

# Check logs
docker logs -f model-api

# Test API
curl http://localhost:8000/health

# Start all services with docker-compose
docker-compose up -d

# Scale API service (multiple instances)
# WHY: Load balancer distributes traffic across instances
docker-compose up -d --scale model-api=3

# View logs
docker-compose logs -f model-api

# Stop all services
docker-compose down
```

**Container image sizes:**

| Stage | Size | Contents |
|-------|------|----------|
| Full build | 2.5 GB | Python + build tools + deps + model |
| Multi-stage | 800 MB | Python + runtime deps + model |
| Optimized | 400 MB | Minimal Python + deps + model |
| Savings | **84%** | From 2.5 GB → 400 MB |

---

## Part 7: Framework Selection Guide

### Decision Matrix

```python
# framework_selector.py
from enum import Enum
from typing import List

class Requirement(Enum):
    FLEXIBILITY = "flexibility"          # Custom preprocessing, business logic
    BATTERIES_INCLUDED = "batteries"     # Minimal setup, built-in features
    LOW_LATENCY = "low_latency"         # < 10ms target
    CROSS_FRAMEWORK = "cross_framework"  # PyTorch + TensorFlow support
    EDGE_DEPLOYMENT = "edge"            # Mobile, IoT, embedded
    EASE_OF_DEBUG = "debug"             # Development experience
    HIGH_THROUGHPUT = "throughput"      # > 1000 req/sec

class Framework(Enum):
    FASTAPI = "fastapi"
    TORCHSERVE = "torchserve"
    GRPC = "grpc"
    ONNX = "onnx"

# Framework capabilities (0-5 scale)
FRAMEWORK_SCORES = {
    Framework.FASTAPI: {
        Requirement.FLEXIBILITY: 5,        # Full control
        Requirement.BATTERIES_INCLUDED: 2, # Manual implementation
        Requirement.LOW_LATENCY: 3,        # 10-20ms
        Requirement.CROSS_FRAMEWORK: 4,    # Any Python model
        Requirement.EDGE_DEPLOYMENT: 2,    # Heavyweight
        Requirement.EASE_OF_DEBUG: 5,      # Excellent debugging
        Requirement.HIGH_THROUGHPUT: 3     # 100-500 req/sec
    },
    Framework.TORCHSERVE: {
        Requirement.FLEXIBILITY: 3,        # Customizable via handlers
        Requirement.BATTERIES_INCLUDED: 5, # Everything built-in
        Requirement.LOW_LATENCY: 4,        # 5-15ms
        Requirement.CROSS_FRAMEWORK: 1,    # PyTorch only
        Requirement.EDGE_DEPLOYMENT: 2,    # Heavyweight
        Requirement.EASE_OF_DEBUG: 3,      # Learning curve
        Requirement.HIGH_THROUGHPUT: 5     # 1000+ req/sec with batching
    },
    Framework.GRPC: {
        Requirement.FLEXIBILITY: 4,        # Binary protocol, custom logic
        Requirement.BATTERIES_INCLUDED: 2, # Manual implementation
        Requirement.LOW_LATENCY: 5,        # 3-8ms
        Requirement.CROSS_FRAMEWORK: 4,    # Any model
        Requirement.EDGE_DEPLOYMENT: 3,    # Moderate size
        Requirement.EASE_OF_DEBUG: 2,      # Binary protocol harder
        Requirement.HIGH_THROUGHPUT: 5     # 1000+ req/sec
    },
    Framework.ONNX: {
        Requirement.FLEXIBILITY: 3,        # Limited to ONNX ops
        Requirement.BATTERIES_INCLUDED: 3, # Runtime provided
        Requirement.LOW_LATENCY: 5,        # 2-6ms (optimized)
        Requirement.CROSS_FRAMEWORK: 5,    # Any framework → ONNX
        Requirement.EDGE_DEPLOYMENT: 5,    # Lightweight runtime
        Requirement.EASE_OF_DEBUG: 2,      # Conversion can be tricky
        Requirement.HIGH_THROUGHPUT: 4     # 500-1000 req/sec
    }
}

def select_framework(
    requirements: List[Requirement],
    weights: List[float] = None
) -> Framework:
    """
    Select best framework based on requirements.

    Args:
        requirements: List of requirements
        weights: Importance weight for each requirement (0-1)

    Returns:
        Best framework
    """
    if weights is None:
        weights = [1.0] * len(requirements)

    scores = {}

    for framework in Framework:
        score = 0
        for req, weight in zip(requirements, weights):
            score += FRAMEWORK_SCORES[framework][req] * weight
        scores[framework] = score

    best_framework = max(scores, key=scores.get)

    print(f"\nFramework Selection:")
    print(f"Requirements: {[r.value for r in requirements]}")
    print(f"\nScores:")
    for framework, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {framework.value}: {score:.1f}")

    return best_framework

# Example use cases
print("=" * 60)
print("Use Case 1: Prototyping with flexibility")
print("=" * 60)
selected = select_framework([
    Requirement.FLEXIBILITY,
    Requirement.EASE_OF_DEBUG
])
print(f"\nRecommendation: {selected.value}")
# Expected: FASTAPI

print("\n" + "=" * 60)
print("Use Case 2: Production PyTorch with minimal setup")
print("=" * 60)
selected = select_framework([
    Requirement.BATTERIES_INCLUDED,
    Requirement.HIGH_THROUGHPUT
])
print(f"\nRecommendation: {selected.value}")
# Expected: TORCHSERVE

print("\n" + "=" * 60)
print("Use Case 3: Low-latency microservice")
print("=" * 60)
selected = select_framework([
    Requirement.LOW_LATENCY,
    Requirement.HIGH_THROUGHPUT
])
print(f"\nRecommendation: {selected.value}")
# Expected: GRPC or ONNX

print("\n" + "=" * 60)
print("Use Case 4: Edge deployment (mobile/IoT)")
print("=" * 60)
selected = select_framework([
    Requirement.EDGE_DEPLOYMENT,
    Requirement.CROSS_FRAMEWORK,
    Requirement.LOW_LATENCY
])
print(f"\nRecommendation: {selected.value}")
# Expected: ONNX

print("\n" + "=" * 60)
print("Use Case 5: Multi-framework ML platform")
print("=" * 60)
selected = select_framework([
    Requirement.CROSS_FRAMEWORK,
    Requirement.HIGH_THROUGHPUT,
    Requirement.BATTERIES_INCLUDED
])
print(f"\nRecommendation: {selected.value}")
# Expected: ONNX or TORCHSERVE (depending on weights)
```

### Quick Reference Guide

| Scenario | Framework | Why |
|----------|-----------|-----|
| **Prototyping** | FastAPI | Fast iteration, easy debugging |
| **PyTorch production** | TorchServe | Built-in batching, metrics, management |
| **Internal microservices** | gRPC | Lowest latency, high throughput |
| **Multi-framework** | ONNX Runtime | Framework-agnostic, optimized |
| **Edge/mobile** | ONNX Runtime | Lightweight, cross-platform |
| **Custom preprocessing** | FastAPI | Full flexibility |
| **High throughput batch** | TorchServe + batching | Dynamic batching built-in |
| **Real-time streaming** | gRPC | Bidirectional streaming |

---

## Summary

**Model serving is pattern matching, not one-size-fits-all.**

**Core patterns:**
1. **FastAPI:** Flexibility, custom logic, easy debugging
2. **TorchServe:** PyTorch batteries-included, built-in batching
3. **gRPC:** Low latency (3-5ms), high throughput, microservices
4. **ONNX Runtime:** Cross-framework, optimized, edge deployment
5. **Dynamic batching:** Adaptive batch size, balances latency and throughput
6. **Containerization:** Reproducible, portable, scalable

**Selection checklist:**
- ✓ Identify primary requirement (flexibility, latency, throughput, etc.)
- ✓ Match requirement to framework strengths
- ✓ Consider deployment environment (cloud, edge, on-prem)
- ✓ Evaluate trade-offs (development speed vs performance)
- ✓ Implement batching if GPU-based (10× better utilization)
- ✓ Containerize for reproducibility
- ✓ Monitor metrics (latency, throughput, GPU util)
- ✓ Iterate based on production data

**Anti-patterns to avoid:**
- ✗ model.pkl in repo (dependency hell)
- ✗ gRPC for simple REST use cases (over-engineering)
- ✗ No batching with GPU (wasted 80% capacity)
- ✗ Not containerized (deployment inconsistency)
- ✗ Static batching (poor latency at low load)

Production-ready model serving requires matching infrastructure pattern to requirements.
