
# Model Serving Patterns Skill

## When to Use This Skill

Use this skill when:
- Deploying ML models to production environments
- Building model serving APIs for real-time inference
- Optimizing model serving for throughput and latency
- Containerizing models for consistent deployment
- Implementing request batching for efficiency
- Choosing between serving frameworks and protocols
- Selecting an LLM serving stack (vLLM, SGLang, TensorRT-LLM, TGI, etc.)

**When NOT to use:** Notebook prototyping, training jobs, or single-prediction scripts where serving infrastructure is premature.

**For LLM-specific generation-quality and inference-optimization tradeoffs** (sampling, KV cache sizing, speculative-decoding hyperparameter tuning, prompt design): see `yzmir-llm-specialist/llm-inference-optimization.md`. This sheet owns the **serving-stack ops** view (which engine, what hardware, batching/caching/distribution mechanics); the LLM-specialist sheet owns the **generation-quality tuning** view. The cross-reference is bidirectional.

## Core Principle

**Serving infrastructure is not one-size-fits-all. Pattern selection is context-dependent.**

Without proper serving infrastructure:
- model.pkl in repo (manual dependency hell)
- Wrong protocol choice (gRPC for simple REST use cases)
- No batching (1 req/sec instead of 100 req/sec)
- Not containerized (works on my machine syndrome)
- Static batching when dynamic needed (underutilized GPU)
- Generic web framework where an LLM-aware engine would give 5-10× more throughput

**Formula:** Right framework (FastAPI vs gRPC vs ONNX vs vLLM/SGLang/TensorRT-LLM) + Request batching (continuous > dynamic > static) + Containerization + Clear selection criteria = Production-ready serving.

## Serving Framework Decision Tree

```
┌────────────────────────────────────────┐
│   Are you serving an LLM?              │
└──────────────┬─────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
       YES            NO
        │             │
        ▼             ▼
    LLM stack     General-purpose stack
    (Part 8):     (Parts 1-4):
    vLLM,         FastAPI / gRPC /
    SGLang,       ONNX / Triton /
    TensorRT-LLM, Ray Serve / BentoML
    TGI, etc.
                       │
                       ▼
               Add request batching
               (Part 5: dynamic for non-LLM,
                continuous for LLM)
                       │
                       ▼
               Containerize (Part 6)
                       │
                       ▼
               Pick framework (Part 7
               selection matrix)
```

## Part 1: FastAPI for Custom Serving

**When to use:** Need flexibility, custom preprocessing, or non-standard workflows. Good default for non-LLM models with simple request/response shapes.

**Advantages:** Full control, easy debugging, Python ecosystem integration.
**Disadvantages:** Manual optimization, no built-in model management.

### Basic FastAPI Serving (modern lifespan API)

```python
# serve_fastapi.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request schema with validation."""
    inputs: List[List[float]] = Field(..., description="Input features as 2D array")
    return_probabilities: bool = Field(False, description="Return class probabilities")


class PredictionResponse(BaseModel):
    """Response schema."""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    latency_ms: float


class ModelServer:
    """
    Model server with explicit load and predict methods.

    WHY: Load model once at startup, reuse across requests.
    WHY: Avoids 5-10 second model loading per request.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info("Model loaded successfully")

    def unload_model(self):
        """Release GPU memory on shutdown."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(self, inputs: np.ndarray):
        self.load_model()
        x = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        # WHY: torch.no_grad() disables gradient computation for inference
        # WHY: Reduces memory usage by ~50% and speeds up by ~2×
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions.cpu().numpy(), probabilities.cpu().numpy()


# Single shared instance
model_server = ModelServer(model_path="model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI startup/shutdown using a single async context manager.

    WHY: FastAPI ≥0.93 deprecated @app.on_event("startup") /
    @app.on_event("shutdown"). The lifespan handler runs setup before
    `yield` and teardown after, with one shared scope (so resources
    initialised at startup are visible to the shutdown branch).
    See https://fastapi.tiangolo.com/advanced/events/
    """
    model_server.load_model()
    logger.info("Server startup complete")
    yield
    # Shutdown
    logger.info("Server shutting down")
    model_server.unload_model()


app = FastAPI(title="Model Serving API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None,
        "device": str(model_server.device),
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
        if inputs.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D array, got {inputs.ndim}D",
            )

        predictions, probabilities = model_server.predict(inputs)
        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist() if request.return_probabilities else None,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000 --workers 4
```

**Performance characteristics (rule-of-thumb, single worker, modern hardware):**

| Metric | Value | Notes |
|--------|-------|-------|
| Cold start | 5-10s | Model loading time |
| Warm latency | 10-50ms | Per request |
| Throughput | 100-500 req/sec | Single worker |
| Memory | 2-8GB | Model + runtime |

### Advanced: Async FastAPI with Background Batching

```python
# serve_fastapi_async.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from asyncio import Queue, create_task, sleep
import asyncio
from typing import Dict
import uuid
import numpy as np


class AsyncBatchPredictor:
    """
    Async batch predictor with request queuing.

    WHY: Collect multiple requests, predict as a batch.
    WHY: GPU utilization: 20% (1 req) → 80% (batch of 32).
    """

    def __init__(self, model_server: "ModelServer", batch_size: int = 32, wait_ms: int = 10):
        self.model_server = model_server
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.queue: Queue = Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = create_task(self._batch_processing_loop())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()

    async def _batch_processing_loop(self):
        while True:
            batch_requests, batch_ids = [], []
            deadline = asyncio.get_event_loop().time() + (self.wait_ms / 1000)

            while len(batch_requests) < self.batch_size:
                timeout = max(0, deadline - asyncio.get_event_loop().time())
                try:
                    request_id, inputs = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    batch_requests.append(inputs)
                    batch_ids.append(request_id)
                except asyncio.TimeoutError:
                    break

            if not batch_requests:
                await sleep(0.001)
                continue

            batch_array = np.array(batch_requests)
            predictions, probabilities = self.model_server.predict(batch_array)

            for i, request_id in enumerate(batch_ids):
                future = self.pending_requests.pop(request_id)
                future.set_result((predictions[i], probabilities[i]))

    async def predict_async(self, inputs):
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        await self.queue.put((request_id, inputs))
        return await future


async_predictor: AsyncBatchPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_predictor
    model_server.load_model()
    async_predictor = AsyncBatchPredictor(model_server, batch_size=32, wait_ms=10)
    await async_predictor.start()
    yield
    await async_predictor.stop()
    model_server.unload_model()


app = FastAPI(lifespan=lifespan)


@app.post("/predict_async")
async def predict_async(request: PredictionRequest):
    inputs = request.inputs[0]
    prediction, probability = await async_predictor.predict_async(inputs)
    return {"prediction": int(prediction), "probability": probability.tolist()}
```

**Performance improvement:**

| Approach | Throughput | GPU Utilization | Latency P95 |
|----------|-----------|-----------------|-------------|
| Synchronous | ~100 req/sec | ~20% | ~15ms |
| Async batching | ~1000 req/sec | ~80% | ~25ms |
| Improvement | **10×** | **4×** | +10ms |


## Part 2: TorchServe (Legacy / Maintenance Mode)

**Status:** TorchServe entered limited maintenance in 2024 and the upstream `pytorch/serve` repository was archived in August 2025; there are no planned new features, bug fixes, or security patches ([upstream issue #3396](https://github.com/pytorch/serve/issues/3396)). For new work, prefer Ray Serve, BentoML, NVIDIA Triton Inference Server, or — for LLMs specifically — vLLM / SGLang / TensorRT-LLM (Part 8).

**Migration urgency:** If you have an existing TorchServe deployment that works, you do not need to migrate immediately. Plan migration on your normal lifecycle cadence, prioritizing it ahead of any change that would require a CVE patch from upstream (which won't come).

The walkthrough below is preserved for teams maintaining existing TorchServe deployments.

### Creating a TorchServe Handler (existing deployments)

```python
# handler.py
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import logging

logger = logging.getLogger(__name__)


class CustomClassifierHandler(BaseHandler):
    """
    Custom TorchServe handler with preprocessing and dynamic batching.

    WHY: TorchServe provides model versioning, metrics, and built-in
    dynamic batching. As of 2024-2025 the project is in maintenance
    mode (see Part 2 header); this code is for keeping existing
    deployments running.
    """

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = f"{model_dir}/{serialized_file}"

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.initialized = True

    def preprocess(self, data):
        """TorchServe batches requests automatically; this method receives
        a list of incoming requests as a single batch."""
        inputs = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode("utf-8")
            tensor = torch.tensor(eval(input_data), dtype=torch.float32)
            tensor = (tensor - self.mean) / self.std
            inputs.append(tensor)
        return torch.stack(inputs).to(self.device)

    def inference(self, batch):
        with torch.no_grad():
            return self.model(batch)

    def postprocess(self, inference_output):
        probabilities = F.softmax(inference_output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return [
            {
                "prediction": predictions[i].item(),
                "probabilities": probabilities[i].tolist(),
            }
            for i in range(len(predictions))
        ]
```

### TorchServe Configuration (existing deployments)

```yaml
# model_config.yaml
minWorkers: 2
maxWorkers: 4
batchSize: 32          # Maximum batch size
maxBatchDelay: 10      # Max wait time for batch (ms)
responseTimeout: 120   # Request timeout (s)
deviceType: "gpu"
deviceIds: [0]
metrics:
  enable: true
  prometheus: true
```

### Packaging and Serving (existing deployments)

```bash
torch-model-archiver \
  --model-name classifier \
  --version 1.0 \
  --serialized-file model.pt \
  --handler handler.py \
  --extra-files "model_config.yaml" \
  --export-path model_store/

torchserve \
  --start \
  --ncs \
  --model-store model_store \
  --models classifier.mar \
  --ts-config config.properties

curl -X POST "http://localhost:8080/predictions/classifier" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0]]}'

curl http://localhost:8082/metrics
```

**For new deployments, evaluate (in this order of typical fit):**

| Use case | Recommended replacement |
|----------|------------------------|
| PyTorch + want batteries-included serving | NVIDIA Triton Inference Server (PyTorch backend) |
| PyTorch + multi-model orchestration | Ray Serve |
| PyTorch + ergonomic packaging | BentoML |
| LLM serving | vLLM / SGLang / TensorRT-LLM (see Part 8) |


## Part 3: gRPC for Low-Latency Serving

**When to use:** Low latency critical (< 10ms), internal services, microservices architecture.

**Advantages:** 3-5× faster than REST, binary protocol (Protocol Buffers + HTTP/2), streaming support.
**Disadvantages:** More complex, requires proto definitions, harder debugging.

### Protocol Definition

```protobuf
// model_service.proto
syntax = "proto3";

package modelserving;

service ModelService {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc PredictStream (PredictRequest) returns (stream PredictResponse);
  rpc PredictBidi (stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  repeated float features = 1;
  bool return_probabilities = 2;
}

message PredictResponse {
  int32 prediction = 1;
  repeated float probabilities = 2;
  float latency_ms = 3;
}

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
        start_time = time.time()
        try:
            features = np.array(request.features, dtype=np.float32)
            x = torch.tensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

            latency_ms = (time.time() - start_time) * 1000
            response = model_service_pb2.PredictResponse(
                prediction=int(pred.item()),
                latency_ms=latency_ms,
            )
            if request.return_probabilities:
                response.probabilities.extend(probs[0].cpu().tolist())
            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.PredictResponse()

    def PredictStream(self, request, context):
        for _ in range(10):
            yield self.Predict(request, context)
            time.sleep(0.01)

    def PredictBidi(self, request_iterator, context):
        for request in request_iterator:
            yield self.Predict(request, context)


class HealthServicer(model_service_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        return model_service_pb2.HealthCheckResponse(
            status=model_service_pb2.HealthCheckResponse.SERVING
        )


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 10 * 1024 * 1024),
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ('grpc.so_reuseport', 1),
            ('grpc.use_local_subchannel_pool', 1),
        ],
    )
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServicer("model.pth"), server
    )
    model_service_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
```

**gRPC vs REST comparison (rule-of-thumb):**

| Metric | gRPC | REST | Advantage |
|--------|------|------|-----------|
| Latency | 3-5ms | 10-15ms | gRPC 3× faster |
| Throughput | ~10k req/sec | ~3k req/sec | gRPC 3× higher |
| Payload size | Binary (smaller) | JSON (larger) | gRPC 30-50% smaller |
| Debugging | Harder | Easier | REST |
| Browser support | No (requires gRPC-Web proxy) | Yes | REST |
| Streaming | Native (server / client / bidi) | Complex (SSE/WebSocket) | gRPC |


## Part 4: ONNX Runtime for Cross-Framework Serving

**When to use:** Cross-framework support (PyTorch, TensorFlow, etc.), maximum CPU performance, edge devices.

**Advantages:** Framework-agnostic, highly optimized, smaller deployment size.
**Disadvantages:** Not all models convert easily, limited debugging.

### Converting PyTorch to ONNX

```python
# convert_to_onnx.py
import torch
import torch.onnx


def convert_pytorch_to_onnx(model_path: str, output_path: str):
    model = torch.load(model_path)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,         # WHY: Opset 17 is the modern stable target
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model exported to {output_path}; ONNX validation OK")


convert_pytorch_to_onnx("model.pth", "model.onnx")
```

### ONNX Runtime Serving

```python
# serve_onnx.py
import onnxruntime as ort
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)


class ONNXModelServer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load_model(self):
        if self.session is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_options, providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"ONNX model loaded; provider={self.session.get_providers()[0]}")

    def predict(self, inputs: np.ndarray):
        self.load_model()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs.astype(np.float32)},
        )
        return outputs[0]


onnx_server = ONNXModelServer("model.onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    onnx_server.load_model()
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    inputs: List[List[float]]


@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = np.array(request.inputs, dtype=np.float32)
    outputs = onnx_server.predict(inputs)
    return {"predictions": outputs.tolist()}
```

**ONNX Runtime advantages:**

| Feature | Benefit | Measurement |
|---------|---------|-------------|
| Speed | Optimized operators | 2-3× faster than naive PyTorch on CPU |
| Size | No framework dependency | tens of MB vs hundreds (PyTorch wheel) |
| Portability | Framework-agnostic | PyTorch / TF / JAX → ONNX |
| Edge deployment | Lightweight runtime | Mobile, IoT, embedded |


## Part 5: Request Batching Patterns

**Core principle:** Batch requests for accelerator efficiency.

**Why batching matters (non-LLM):**
- GPU utilization: 20% (single request) → 80% (batch of 32)
- Throughput: 100 req/sec → 1000 req/sec
- Cost: roughly 10× reduction in $/req on GPU

There are three batching regimes; pick the one that matches your model class:

| Regime | What it is | Where it fits |
|--------|------------|--------------|
| **Static batching** | Wait until N requests arrive (or fixed window), then run | Offline batch jobs only — bad latency at low load |
| **Dynamic batching** | Wait up to T ms or until N requests arrive, then run; configurable | Most non-LLM models (Triton, TorchServe, custom FastAPI) |
| **Continuous batching (in-flight batching)** | Insert and evict requests at every decoding step; per-request lengths can differ | LLM serving — see 5.2 |

### 5.1 Dynamic Batching (non-LLM)

```python
# dynamic_batching.py
import asyncio
from asyncio import Queue
from typing import List, Tuple
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class DynamicBatcher:
    """
    Dynamic batching with adaptive timeout.

    WHY: Static batching waits for full batch (high latency at low load).
    WHY: Dynamic batching adapts: full batch OR timeout.
    """

    def __init__(self, model_server, max_batch_size: int = 32, max_wait_ms: int = 10):
        self.model_server = model_server
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue: Queue = Queue()
        self.stats = {"total_requests": 0, "total_batches": 0,
                      "avg_batch_size": 0, "gpu_utilization": 0}

    async def start(self):
        asyncio.create_task(self._batch_loop())

    async def _batch_loop(self):
        while True:
            batch, futures = [], []
            request_data, future = await self.request_queue.get()
            batch.append(request_data)
            futures.append(future)

            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000)
            while len(batch) < self.max_batch_size:
                remaining_time = max(0, deadline - asyncio.get_event_loop().time())
                try:
                    request_data, future = await asyncio.wait_for(
                        self.request_queue.get(), timeout=remaining_time
                    )
                    batch.append(request_data)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch, futures)

    async def _process_batch(self, batch: List[np.ndarray], futures: List[asyncio.Future]):
        batch_size = len(batch)
        batch_array = np.array(batch)
        start_time = time.time()
        predictions, probabilities = self.model_server.predict(batch_array)
        inference_time = (time.time() - start_time) * 1000

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

        for i, future in enumerate(futures):
            if not future.done():
                future.set_result((predictions[i], probabilities[i]))

    async def predict(self, inputs: np.ndarray) -> Tuple[int, np.ndarray]:
        future = asyncio.Future()
        await self.request_queue.put((inputs, future))
        prediction, probability = await future
        return prediction, probability
```

**Dynamic batching performance (rule-of-thumb):**

| Load | Batch Size | GPU Util | Latency | Throughput |
|------|-----------|----------|---------|------------|
| High (100 req/sec) | 28-32 | ~90% | ~12ms | ~1000 req/sec |
| Medium (20 req/sec) | 8-12 | ~35% | ~11ms | ~200 req/sec |
| Low (5 req/sec) | 1-2 | ~10% | ~10ms | ~50 req/sec |

### 5.2 Continuous Batching (LLM-specific)

Dynamic batching is wrong for LLMs because LLM requests do not all finish at the same step. With dynamic batching, the batch stalls until the longest request finishes, wasting most of the accelerator.

**Continuous batching** (also called **in-flight batching**) operates per **decoding step** rather than per **request**: at every token-generation step, finished sequences are evicted and new sequences are spliced in. This was popularized by Orca (OSDI 2022) and made memory-efficient by vLLM's PagedAttention (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)).

vLLM, SGLang, TensorRT-LLM, TGI, and LMDeploy all implement continuous batching natively. **You should not implement this yourself** for any modern LLM workload — use one of the engines in Part 8.

### 5.3 Prefix Caching (engine-level KV-cache reuse)

When two requests share a prefix (system prompt, few-shot examples, document context), the KV cache for that prefix can be computed once and reused, eliminating prefill cost on the shared portion.

- **vLLM** has automatic prefix caching ([vLLM docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)).
- **SGLang** generalizes this with **RadixAttention** (Zheng et al., [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)), which keeps shared prefixes in a radix tree of KV-cache blocks and allows automatic reuse across arbitrarily-shaped request tree fan-outs (agents, branching, structured generation).
- **TensorRT-LLM** supports KV-cache reuse with prefix matching.

This is the **engine-layer** prefix cache — it sits inside the inference server and is invisible to the client. It is **not the same** as the application-layer / API-vendor prompt cache (Anthropic prompt caching, OpenAI prompt caching), which is configured by the API client to skip re-billing/re-processing of repeated prompt prefixes. For the application-layer story see `yzmir-llm-specialist/context-engineering-and-prompt-caching.md`. The two layers compose: an Anthropic-style cache hit avoids the request entirely; an engine-layer prefix cache hit reuses KV memory for requests that did reach the engine.

### 5.4 Speculative Decoding (conceptual)

Standard autoregressive decoding generates one token per forward pass. Speculative decoding produces several candidate tokens cheaply (with a smaller "draft" model or a head attached to the target model) and then verifies them with the full target model in a single pass. Verified tokens are accepted; rejected tokens fall back to the target model's output. Net effect: 1.5-3× speedup on memory-bandwidth-bound decode with no quality change (it is mathematically lossless when implemented correctly).

Engine implementations:
- **Medusa** ([Cai et al., 2024](https://arxiv.org/abs/2401.10774)) — adds extra decoding heads to the target model.
- **EAGLE / EAGLE-2** ([Li et al., 2024](https://arxiv.org/abs/2401.15077)) — draft via a lightweight feature-level autoregressive head; tighter accept rate than vanilla draft-target.
- **Draft-target** — separate small draft model, used by vLLM, TensorRT-LLM, TGI.

For hyperparameter tuning (draft length, acceptance threshold, draft-target pairing): see `yzmir-llm-specialist/llm-inference-optimization.md`.


## Part 6: Containerization

**Why containerize:** "Works on my machine" → "Works everywhere"

**Benefits:**
- Reproducible builds (same dependencies, versions)
- Isolated environment (no conflicts)
- Portable deployment (dev, staging, prod identical)
- Easy scaling (Kubernetes, Docker Swarm)

### Multi-Stage Docker Build

```dockerfile
# Dockerfile
# WHY: Multi-stage build reduces image size by 50-80%

# ==================== Stage 1: Build ====================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime ====================
FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY serve_fastapi.py .
COPY model.pth .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Multi-Service

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pth
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

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

```bash
docker build -t model-api:1.0.0 .
docker run -d --name model-api -p 8000:8000 --gpus all model-api:1.0.0
docker compose up -d
docker compose up -d --scale model-api=3
```

**Container image sizes (illustrative):**

| Stage | Size | Contents |
|-------|------|----------|
| Full build | ~2.5 GB | Python + build tools + deps + model |
| Multi-stage | ~800 MB | Python + runtime deps + model |
| Optimized | ~400 MB | Minimal Python + deps + model |


## Part 7: General-Purpose Framework Selection (non-LLM)

### Decision Matrix

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Prototyping / custom logic | FastAPI | Fast iteration, easy debugging |
| Multi-framework production (PyTorch/TF/ONNX) | NVIDIA Triton Inference Server | Multi-backend, dynamic batching, model versioning |
| Multi-model orchestration / pipelines | Ray Serve | DAG composition, autoscaling, deployment graphs |
| Ergonomic packaging + multi-engine | BentoML | Service abstraction, integrates with vLLM, Triton, etc. |
| Internal microservice, low latency | gRPC | 3-5ms latency, binary protocol |
| Multi-framework / edge | ONNX Runtime | Lightweight, optimized, portable |
| Custom preprocessing | FastAPI | Full flexibility |
| Real-time streaming | gRPC | Bidirectional streaming |
| Existing PyTorch deployment | TorchServe (maintenance only) | Don't migrate just for the sake of it |

### Quick characteristics

| Framework | Strengths | Weaknesses | Status |
|-----------|-----------|-----------|--------|
| **FastAPI** | Flexibility, debugging, Python-native | Manual batching/metrics | Active |
| **NVIDIA Triton Inference Server** | Multi-backend (PyTorch/TF/ONNX/TensorRT/Python), dynamic batching, ensemble, GPU sharing, prod-grade metrics | Heavier ops surface | Active |
| **Ray Serve** | Multi-model graphs, autoscaling, fits Ray ecosystem | Cluster overhead for simple cases | Active |
| **BentoML** | Pythonic packaging, multi-engine integration (vLLM, Triton, TGI) | Less raw performance than purpose-built engines | Active |
| **gRPC (custom)** | Lowest network latency | Proto overhead, harder debugging | Active |
| **ONNX Runtime (custom)** | Cross-framework, edge | Conversion can be tricky | Active |
| **TorchServe** | Built-in batching, model versioning | Maintenance mode (Aug 2025) | Maintenance |


## Part 8: LLM Serving Stacks

LLMs change the serving question because:
- Decode is memory-bandwidth bound, not compute bound — batching tokens, not requests, is what matters.
- KV cache memory is the binding constraint, not weights.
- Requests have varying lengths; per-step scheduling beats per-request scheduling.
- The OpenAI-compatible HTTP surface (`/v1/chat/completions`, `/v1/completions`, function/tool calling) is now the de-facto contract clients expect.

The right tool depends on hardware and goals.

### 8.1 Engine descriptions

**vLLM**
Default open-source choice for self-hosted LLM serving. PagedAttention KV cache (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)), continuous batching, automatic prefix caching, speculative decoding, tensor and pipeline parallelism, OpenAI-compatible API, broad hardware support (NVIDIA, AMD ROCm, Intel CPU/GPU, AWS Neuron, Google TPU). Best general default. ([Project](https://github.com/vllm-project/vllm))

**SGLang**
RadixAttention (Zheng et al., [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) — automatic KV-cache reuse over a radix tree of prefixes, which dominates on tree-shaped workloads (agent loops, branching, structured generation, RAG with shared system prompts). Compressed-FSM constrained decoding for fast structured output (JSON-schema, regex). Strong choice when many requests share prefixes or you need high-throughput structured output. ([Project](https://github.com/sgl-project/sglang))

**NVIDIA TensorRT-LLM**
Highest single-GPU throughput on NVIDIA hardware (Hopper/Ada/Blackwell), with native FP8 and MXFP4 support, in-flight batching, kernel fusion, paged KV cache. Build process is heavier than vLLM (engines must be compiled per model + GPU + precision combination), but raw performance ceiling is highest on NVIDIA. ([Project](https://github.com/NVIDIA/TensorRT-LLM))

**NVIDIA Triton Inference Server (with TensorRT-LLM backend)**
Triton is the production server; the TensorRT-LLM backend gives Triton native LLM-aware batching and KV management. Pick this when you also serve non-LLM models (vision, embedding, ranking) from the same fleet and want one ops surface. ([Triton project](https://github.com/triton-inference-server/server))

**Hugging Face TGI (Text Generation Inference)**
Production server from Hugging Face; tight integration with the Hub (any HF causal-LM checkpoint), continuous batching, speculative decoding, quantization integrations (AWQ/GPTQ/bitsandbytes/EETQ/FP8), broad hardware (NVIDIA, AMD ROCm, Intel Gaudi, AWS Inferentia). Easy default if your team is already in the HF ecosystem. ([Project](https://github.com/huggingface/text-generation-inference))

**LMDeploy**
InternLM team. TurboMind backend (NVIDIA, custom CUDA kernels) and PyTorch backend; strong support for the InternLM and Qwen model families; INT4/INT8/FP8 weight quant; image/video LLM support. Often best perf for InternLM/Qwen models specifically. ([Project](https://github.com/InternLM/lmdeploy))

**Ray Serve**
Not LLM-specialized; use it as a multi-model orchestration layer over vLLM or other engines. Natural fit when an LLM is one node in a larger graph (retriever → reranker → LLM → tool) and you need autoscaling. ([Project](https://docs.ray.io/en/latest/serve/index.html))

**BentoML (with vLLM / TGI / etc.)**
Ergonomic packaging and deployment of services that wrap a serving engine. Best when you want a single artifact ("Bento") that pins a vLLM/TGI/TensorRT-LLM engine plus pre/post-processing and ships through your deploy pipeline. ([Project](https://github.com/bentoml/BentoML))

**llama.cpp + ExLlamaV2**
The CPU / laptop / Apple Silicon path. llama.cpp is the canonical engine for GGUF k-quants; ExLlamaV2 targets NVIDIA GPUs with very efficient INT4 (EXL2) kernels, popular on consumer hardware. Use these for local development, on-device, or when you don't have a server-class GPU. ([llama.cpp](https://github.com/ggml-org/llama.cpp), [ExLlamaV2](https://github.com/turboderp/exllamav2))

**MLC-LLM**
Universal deployment via TVM compilation: targets browsers (WebGPU), iOS, Android, Mac/Linux/Windows desktop, NVIDIA, AMD, Apple, Qualcomm. Use when you need a single model artifact that runs across mobile/web/desktop. ([Project](https://github.com/mlc-ai/mlc-llm))

### 8.2 LLM stack selection matrix

| Goal / constraint | Recommended primary | Notes |
|-------------------|--------------------|-------|
| **Default self-hosted serving** | vLLM | Easiest path to OpenAI-API-compatible endpoint |
| **Highest throughput on NVIDIA** | TensorRT-LLM (or Triton + TRT-LLM) | Build complexity is the cost |
| **Many requests sharing system prompt / agent fan-out / structured output** | SGLang | RadixAttention + compressed-FSM decoding |
| **One fleet, mixed LLM + non-LLM models** | Triton (+ TRT-LLM backend for LLMs) | Single ops surface |
| **Tight HF Hub integration / "just point at a checkpoint"** | TGI | Best HF ecosystem fit |
| **InternLM / Qwen models** | LMDeploy | TurboMind kernels are tuned for these |
| **CPU / laptop / Apple Silicon** | llama.cpp | GGUF k-quants are the standard format |
| **Consumer NVIDIA GPU, INT4** | ExLlamaV2 | EXL2 is very fast on a single 4090 |
| **Mobile / browser / cross-platform** | MLC-LLM | One pipeline, many targets |
| **Multi-model graph (retriever + reranker + LLM)** | Ray Serve over vLLM | Compose engines |
| **Need ergonomic packaging + Ops** | BentoML wrapping vLLM/TGI | Ship a Bento |

### 8.3 Decision-axis cross-cuts

| Axis | vLLM | SGLang | TensorRT-LLM | TGI | LMDeploy | llama.cpp / EXL2 | MLC-LLM |
|------|------|--------|---------------|-----|----------|------------------|---------|
| Throughput-vs-latency emphasis | high throughput, balanced latency | high throughput on shared-prefix workloads | highest throughput on NVIDIA | balanced | balanced | latency on small hardware | balanced |
| Single-GPU | yes | yes | yes | yes | yes | yes | yes |
| Multi-GPU (TP/PP) | yes | yes | yes | yes | yes | partial (TP) | partial |
| Distributed multi-node | yes (via Ray) | yes | yes | yes | partial | no | no |
| OpenAI-compatible API | yes | yes | yes (via Triton + frontends) | yes | yes | yes (via project's server) | yes |
| Hardware target | NVIDIA, AMD, Intel, TPU, Neuron | NVIDIA, AMD | NVIDIA only | NVIDIA, AMD, Gaudi, Inferentia | NVIDIA | CPU, NVIDIA, Apple, AMD | mobile, web, NVIDIA, Apple, AMD, Qualcomm |
| Best quantization story | AWQ/GPTQ/FP8/bnb | AWQ/GPTQ/FP8 | FP8/INT4/MXFP4 (Blackwell) | AWQ/GPTQ/FP8/bnb | INT4/INT8/FP8 (turbomind) | GGUF k-quants / EXL2 | per-target |

### 8.4 Cross-references

- For LLM-side optimization decisions (which quantization for *this* model, KV-cache sizing, speculative-decoding hyperparameters, sampling): `yzmir-llm-specialist/llm-inference-optimization.md`.
- For application-layer prompt caching (Anthropic prompt cache, OpenAI prompt cache): `yzmir-llm-specialist/context-engineering-and-prompt-caching.md`.
- For agent / tool-use serving patterns and MCP integration: `yzmir-llm-specialist/agentic-patterns-and-mcp.md`.
- For RAG architecture (retriever + reranker + LLM compositions you'd run behind Ray Serve): `yzmir-llm-specialist/rag-architecture-patterns.md`.


## Part 9: Cloud Managed Inference Services

When you don't want to operate the serving stack yourself.

| Service | Notes |
|---------|-------|
| **AWS SageMaker real-time endpoints** | Managed model hosting, autoscaling, supports LMI (Large Model Inference) container with vLLM / TGI / TensorRT-LLM under the hood |
| **AWS Inferentia / Trainium** (Inf2 / Trn2 instances) | Custom silicon; access via SageMaker, EC2, or EKS; supported by vLLM and TGI via Neuron SDK |
| **GCP Vertex AI Prediction / Vertex AI Endpoints** | Managed online prediction; supports custom containers and pre-built model garden deployments |
| **GCP Cloud TPU v5e/v5p inference** | TPU inference via JAX/PyTorch-XLA; vLLM has TPU support |
| **Azure ML online endpoints** | Managed endpoints with autoscaling, blue/green deployments |
| **Modal** | Serverless GPU; sub-second cold starts via image snapshots; ergonomic Python API |
| **Replicate** | Hosted inference for community models; pay-per-second; OpenAI-style API |
| **Together AI** | Hosted open-weight LLM inference with OpenAI-compatible API |
| **Anyscale** | Managed Ray (and Ray Serve) for LLM and ML serving |
| **Fireworks / DeepInfra / Lepton / RunPod / Cerebrium** | Other managed-inference providers; evaluate by latency, cost-per-million-tokens, and OpenAI-API compatibility |

Use these as the inference layer when ops capacity is the binding constraint, and self-host (Parts 7-8) when sustained $/req or data residency is.


## Summary

**Model serving is pattern matching, not one-size-fits-all.**

**Core patterns:**
1. **FastAPI:** Flexibility, custom logic, easy debugging — modern lifespan API, not deprecated `on_event`.
2. **gRPC:** Low latency (3-5ms), high throughput, microservices.
3. **ONNX Runtime:** Cross-framework, optimized, edge deployment.
4. **NVIDIA Triton / Ray Serve / BentoML:** Production multi-model serving — the contemporary replacement for TorchServe (which is in maintenance).
5. **vLLM / SGLang / TensorRT-LLM / TGI / LMDeploy:** LLM-aware engines with continuous batching, paged/radix KV cache, prefix caching, speculative decoding.
6. **Continuous batching for LLMs, dynamic batching for everything else.**
7. **Containerization** for reproducibility.

**Selection checklist:**
- [ ] Identify primary requirement (LLM vs not, latency, throughput, edge)
- [ ] If LLM: pick from Part 8 by hardware and workload shape
- [ ] If not LLM: pick from Part 7 by integration / ops constraints
- [ ] Implement appropriate batching (continuous for LLMs, dynamic otherwise)
- [ ] Containerize
- [ ] Monitor (latency p50/p95/p99, throughput, GPU util, KV-cache hit rate for LLMs)
- [ ] Cross-reference llm-specialist for LLM-side decisions

**Anti-patterns to avoid:**
- model.pkl in repo (dependency hell)
- Generic FastAPI for an LLM workload (you'll lose 5-10× throughput vs vLLM)
- gRPC for simple REST use cases (over-engineering)
- Dynamic batching for LLMs (continuous batching wins)
- Implementing your own continuous batching (use vLLM/SGLang/TRT-LLM)
- Using deprecated `@app.on_event("startup")` in new FastAPI code (use `lifespan`)
- Picking TorchServe for new deployments (maintenance mode)
- Static batching outside offline jobs

Production-ready model serving is matching the right engine to the workload, with the right batching/caching regime, on the right hardware.

---

Tooling and APIs current as of 2026-05; revisit quarterly.
