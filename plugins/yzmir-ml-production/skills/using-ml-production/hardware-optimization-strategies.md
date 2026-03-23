
# Hardware Optimization Strategies

## Overview

This skill provides systematic methodology for optimizing ML model inference performance on specific hardware platforms. Covers GPU optimization (CUDA, TensorRT), CPU optimization (threading, SIMD), and edge optimization (ARM, quantization), with emphasis on profiling-driven optimization and hardware-appropriate technique selection.

**Core Principle**: Profile first to identify bottlenecks, then apply hardware-specific optimizations. Different hardware requires different optimization strategies - GPU benefits from batch size and operator fusion, CPU from threading and SIMD, edge devices from quantization and model architecture.

## When to Use

Use this skill when:
- Model inference performance depends on hardware utilization (not just model architecture)
- Need to optimize for specific hardware: NVIDIA GPU, Intel/AMD CPU, ARM edge devices
- Model is serving bottleneck after profiling (vs data loading, preprocessing)
- Want to maximize throughput or minimize latency on given hardware
- Deploying to resource-constrained edge devices
- User mentions: "optimize for GPU", "CPU inference slow", "edge deployment", "TensorRT", "ONNX Runtime", "batch size tuning"

**Don't use for**:
- Training optimization → use `training-optimization` pack
- Model architecture selection → use `neural-architectures`
- Model compression (pruning, distillation) → use `model-compression-techniques`
- Quantization specifically → use `quantization-for-inference`
- Serving infrastructure → use `model-serving-patterns`

**Boundary with quantization-for-inference**:
- This skill covers hardware-aware quantization deployment (INT8 on CPU vs GPU, ARM NEON)
- `quantization-for-inference` covers quantization techniques (PTQ, QAT, calibration)
- Use both when quantization is part of hardware optimization strategy

## Core Methodology

### Step 1: Profile to Identify Bottlenecks

**ALWAYS profile before optimizing**. Don't guess where time is spent.

#### PyTorch Profiler (Comprehensive)

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

model = load_model().cuda().eval()

# Profile inference
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("inference"):
        with torch.no_grad():
            output = model(input_tensor.cuda())

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Export for visualization
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

**What to look for**:
- **CPU time high**: Data preprocessing, Python overhead, CPU-bound ops
- **CUDA time high**: Model compute is bottleneck, optimize model inference
- **Memory**: Check for out-of-memory issues or unnecessary allocations
- **Operator breakdown**: Which layers/ops are slowest?

#### NVIDIA Profiling Tools

```bash
# Nsight Systems - high-level timeline
nsys profile -o output python inference.py

# Nsight Compute - kernel-level profiling
ncu --set full -o kernel_profile python inference.py

# Simple nvidia-smi monitoring
nvidia-smi dmon -s u -i 0  # Monitor GPU utilization
```

#### Intel VTune (CPU profiling)

```bash
# Profile CPU bottlenecks
vtune -collect hotspots -r vtune_results -- python inference.py

# Analyze results
vtune-gui vtune_results
```

#### Simple Timing

```python
import time
import torch

def profile_pipeline(model, input_data, device='cuda'):
    """Profile each stage of inference pipeline"""

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data.to(device))

    if device == 'cuda':
        torch.cuda.synchronize()  # Critical for accurate GPU timing

    # Profile preprocessing
    t0 = time.time()
    preprocessed = preprocess(input_data)
    t1 = time.time()

    # Profile model inference
    preprocessed = preprocessed.to(device)
    if device == 'cuda':
        torch.cuda.synchronize()

    t2 = time.time()
    with torch.no_grad():
        output = model(preprocessed)

    if device == 'cuda':
        torch.cuda.synchronize()
    t3 = time.time()

    # Profile postprocessing
    result = postprocess(output.cpu())
    t4 = time.time()

    print(f"Preprocessing:  {(t1-t0)*1000:.2f}ms")
    print(f"Model Inference: {(t3-t2)*1000:.2f}ms")
    print(f"Postprocessing: {(t4-t3)*1000:.2f}ms")
    print(f"Total:          {(t4-t0)*1000:.2f}ms")

    return {
        'preprocess': (t1-t0)*1000,
        'inference': (t3-t2)*1000,
        'postprocess': (t4-t3)*1000,
    }
```

**Critical**: Always use `torch.cuda.synchronize()` before timing GPU operations, otherwise you measure kernel launch time, not execution time.


### Step 2: Select Hardware-Appropriate Optimizations

Based on profiling results and target hardware, select appropriate optimization strategies.

## GPU Optimization (NVIDIA CUDA)

### Strategy 1: TensorRT (2-5x Speedup for CNNs/Transformers)

**When to use**:
- NVIDIA GPU (T4, V100, A100, RTX series)
- Model architecture supported (CNN, Transformer, RNN)
- Inference-only workload (not training)
- Want automatic optimization (fusion, precision, kernels)

**Best for**: Production deployment on NVIDIA GPUs, predictable performance gains

```python
import torch
import torch_tensorrt

# Load PyTorch model
model = load_model().eval().cuda()

# Compile to TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 224, 224],    # Minimum batch size
        opt_shape=[8, 3, 224, 224],    # Optimal batch size
        max_shape=[32, 3, 224, 224],   # Maximum batch size
        dtype=torch.float16
    )],
    enabled_precisions={torch.float16},  # Use FP16
    workspace_size=1 << 30,              # 1GB workspace for optimization
    truncate_long_and_double=True
)

# Save compiled model
torch.jit.save(trt_model, "model_trt.ts")

# Inference (same API as PyTorch)
with torch.no_grad():
    output = trt_model(input_tensor.cuda())
```

**What TensorRT does**:
1. **Operator fusion**: Combines conv + bn + relu into single kernel
2. **Precision calibration**: Automatic mixed precision (FP16/INT8)
3. **Kernel auto-tuning**: Selects fastest CUDA kernel for each op
4. **Memory optimization**: Reduces memory transfers
5. **Graph optimization**: Removes unnecessary operations

**Limitations**:
- Only supports NVIDIA GPUs
- Some custom ops may not be supported
- Compilation time (minutes for large models)
- Fixed input shapes (or min/max range)

**Troubleshooting**:
```python
# If compilation fails, try:
# 1. Enable verbose logging
import logging
logging.getLogger("torch_tensorrt").setLevel(logging.DEBUG)

# 2. Disable unsupported layers (fallback to PyTorch)
trt_model = torch_tensorrt.compile(
    model,
    inputs=[...],
    enabled_precisions={torch.float16},
    torch_fallback=torch_tensorrt.TorchFallback()  # Fallback for unsupported ops
)

# 3. Check for unsupported ops
torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Warning)
```


### Strategy 2: torch.compile() (PyTorch 2.0+ - Easy 1.5-2x Speedup)

**When to use**:
- PyTorch 2.0+ available
- Want easy optimization without complexity
- Model has custom operations (TensorRT may not support)
- Rapid prototyping (faster than TensorRT compilation)

**Best for**: Quick wins, development iteration, custom models

```python
import torch

model = load_model().eval().cuda()

# Compile with default backend (inductor)
compiled_model = torch.compile(model)

# Compile with specific mode
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
    fullgraph=True,          # Compile entire graph (vs subgraphs)
)

# First run compiles (slow), subsequent runs are fast
with torch.no_grad():
    output = compiled_model(input_tensor.cuda())
```

**Modes**:
- `default`: Balanced compilation time and runtime performance
- `reduce-overhead`: Minimize Python overhead (best for small models)
- `max-autotune`: Maximum optimization (long compilation, best runtime)

**What torch.compile() does**:
1. **Operator fusion**: Similar to TensorRT
2. **Python overhead reduction**: Removes Python interpreter overhead
3. **Memory optimization**: Reduces allocations
4. **CUDA graph generation**: For fixed-size models

**Advantages over TensorRT**:
- Easier to use (one line of code)
- Supports custom operations
- Faster compilation
- No fixed input shapes

**Disadvantages vs TensorRT**:
- Smaller speedup (1.5-2x vs 2-5x)
- Less mature (newer feature)


### Strategy 3: Mixed Precision (FP16 - Easy 2x Speedup)

**When to use**:
- NVIDIA GPU with Tensor Cores (V100, A100, T4, RTX)
- Model doesn't require FP32 precision
- Want simple optimization (minimal code change)
- Memory-bound models (FP16 uses half the memory)

```python
import torch
from torch.cuda.amp import autocast

model = load_model().eval().cuda().half()  # Convert model to FP16

# Inference with autocast
with torch.no_grad():
    with autocast():
        output = model(input_tensor.cuda().half())
```

**Caution**: Some models lose accuracy with FP16. Test accuracy before deploying.

```python
# Validate FP16 accuracy
def validate_fp16_accuracy(model, test_loader, tolerance=0.01):
    model_fp32 = model.float()
    model_fp16 = model.half()

    diffs = []
    for inputs, _ in test_loader:
        with torch.no_grad():
            output_fp32 = model_fp32(inputs.cuda().float())
            output_fp16 = model_fp16(inputs.cuda().half())

        diff = (output_fp32 - output_fp16.float()).abs().mean().item()
        diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs)
    print(f"Average FP32-FP16 difference: {avg_diff:.6f}")

    if avg_diff > tolerance:
        print(f"WARNING: FP16 accuracy loss exceeds tolerance ({tolerance})")
        return False
    return True
```


### Strategy 4: Batch Size Tuning

**When to use**: Always! Batch size is the most important parameter for GPU throughput.

**Trade-off**:
- **Larger batch** = Higher throughput, higher latency, more memory
- **Smaller batch** = Lower latency, lower throughput, less memory

#### Find Optimal Batch Size

```python
def find_optimal_batch_size(model, input_shape, device='cuda', max_memory_pct=0.9):
    """Binary search for maximum batch size that fits in memory"""
    model = model.to(device).eval()

    # Start with batch size 1, increase until OOM
    batch_size = 1
    max_batch = 1024  # Upper bound

    while batch_size < max_batch:
        try:
            torch.cuda.empty_cache()
            test_batch = torch.randn(batch_size, *input_shape).to(device)

            with torch.no_grad():
                _ = model(test_batch)

            # Check memory usage
            mem_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

            if mem_allocated > max_memory_pct:
                print(f"Batch size {batch_size}: {mem_allocated*100:.1f}% memory (near limit)")
                break

            print(f"Batch size {batch_size}: OK ({mem_allocated*100:.1f}% memory)")
            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                batch_size = batch_size // 2
                break
            else:
                raise e

    print(f"\nOptimal batch size: {batch_size}")
    return batch_size
```

#### Measure Latency vs Throughput

```python
def benchmark_batch_sizes(model, input_shape, batch_sizes=[1, 4, 8, 16, 32, 64], device='cuda', num_runs=100):
    """Measure latency and throughput for different batch sizes"""
    model = model.to(device).eval()
    results = []

    for batch_size in batch_sizes:
        try:
            test_batch = torch.randn(batch_size, *input_shape).to(device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(test_batch)

            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(test_batch)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            latency_per_batch = (elapsed / num_runs) * 1000  # ms
            throughput = (batch_size * num_runs) / elapsed  # samples/sec
            latency_per_sample = latency_per_batch / batch_size  # ms/sample

            results.append({
                'batch_size': batch_size,
                'latency_per_batch_ms': latency_per_batch,
                'latency_per_sample_ms': latency_per_sample,
                'throughput_samples_per_sec': throughput,
            })

            print(f"Batch {batch_size:3d}: {latency_per_batch:6.2f}ms/batch, "
                  f"{latency_per_sample:6.2f}ms/sample, {throughput:8.1f} samples/sec")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size:3d}: OOM")
                break

    return results
```

**Decision criteria**:
- **Online serving (real-time API)**: Use small batch (1-8) for low latency
- **Batch serving**: Use large batch (32-128) for high throughput
- **Dynamic batching**: Let serving framework accumulate requests (TorchServe, Triton)


### Strategy 5: CUDA Graphs (Fixed-Size Inputs - 20-30% Speedup)

**When to use**:
- Fixed input size (no dynamic shapes)
- Small models with many kernel launches
- Already optimized but want last 20% speedup

**What CUDA graphs do**: Record sequence of CUDA operations, replay without CPU overhead

```python
import torch

model = load_model().eval().cuda()

# Static input (fixed size)
static_input = torch.randn(8, 3, 224, 224).cuda()
static_output = torch.randn(8, 1000).cuda()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(static_input)

# Capture graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    with torch.no_grad():
        static_output = model(static_input)

# Replay graph (very fast)
def inference_with_graph(input_tensor):
    # Copy input to static buffer
    static_input.copy_(input_tensor)

    # Replay graph
    graph.replay()

    # Copy output from static buffer
    return static_output.clone()

# Benchmark
input_tensor = torch.randn(8, 3, 224, 224).cuda()
output = inference_with_graph(input_tensor)
```

**Limitations**:
- Fixed input/output shapes (no dynamic batching)
- No control flow (if/else) in model
- Adds complexity (buffer management)


## CPU Optimization (Intel/AMD)

### Strategy 1: Threading Configuration (Critical for Multi-Core)

**When to use**: Always for CPU inference on multi-core machines

**Problem**: PyTorch defaults to 4-8 threads, leaving cores idle

```python
import torch

# Check current configuration
print(f"Intra-op threads: {torch.get_num_threads()}")
print(f"Inter-op threads: {torch.get_num_interop_threads()}")

# Set to number of physical cores (not hyperthreads)
import os
num_cores = os.cpu_count() // 2  # Divide by 2 if hyperthreading enabled

torch.set_num_threads(num_cores)  # Intra-op parallelism (within operations)
torch.set_num_interop_threads(1)  # Inter-op parallelism (between operations, disable to avoid oversubscription)

# Verify
print(f"Set intra-op threads: {torch.get_num_threads()}")
```

**Intra-op vs Inter-op**:
- **Intra-op**: Parallelizes single operation (e.g., matrix multiply uses 32 cores)
- **Inter-op**: Parallelizes independent operations (e.g., run conv1 and conv2 simultaneously)

**Best practice**:
- **Intra-op threads** = number of physical cores (enables each op to use all cores)
- **Inter-op threads** = 1 (disable to avoid oversubscription and context switching)

**Warning**: If using DataLoader with workers, account for those threads:
```python
num_cores = os.cpu_count() // 2
num_dataloader_workers = 4

torch.set_num_threads(num_cores - num_dataloader_workers)  # Leave cores for DataLoader
```


### Strategy 2: MKLDNN/OneDNN Backend (Intel-Optimized Operations)

**When to use**: Intel CPUs (Xeon, Core i7/i9)

**What it does**: Uses Intel's optimized math libraries (AVX, AVX-512)

```python
import torch

# Enable MKLDNN
torch.backends.mkldnn.enabled = True

# Check if available
print(f"MKLDNN available: {torch.backends.mkldnn.is_available()}")

# Inference (automatically uses MKLDNN when beneficial)
model = load_model().eval()
with torch.no_grad():
    output = model(input_tensor)
```

**For maximum performance**: Use channels-last memory format (better cache locality)

```python
model = model.eval()

# Convert to channels-last format (NHWC instead of NCHW)
model = model.to(memory_format=torch.channels_last)

# Input also channels-last
input_tensor = input_tensor.to(memory_format=torch.channels_last)

with torch.no_grad():
    output = model(input_tensor)
```

**Speedup**: 1.5-2x on Intel CPUs with AVX-512


### Strategy 3: ONNX Runtime (Best CPU Performance)

**When to use**:
- Dedicated CPU inference deployment
- Want best possible CPU performance
- Model is fully supported by ONNX

**Advantages**:
- Optimized for CPU (MLAS, DNNL, OpenMP)
- Graph optimizations (fusion, constant folding)
- Quantization support (INT8)

```python
import torch
import onnx
import onnxruntime as ort

# Export PyTorch model to ONNX
model = load_model().eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Optimize ONNX graph
import onnxruntime.transformers.optimizer as optimizer
optimized_model = optimizer.optimize_model("model.onnx", model_type='bert', num_heads=8, hidden_size=512)
optimized_model.save_model_to_file("model_optimized.onnx")

# Create inference session with optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count() // 2
sess_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    "model_optimized.onnx",
    sess_options,
    providers=['CPUExecutionProvider']  # Use CPU
)

# Inference
input_data = input_tensor.numpy()
output = session.run(None, {'input': input_data})[0]
```

**Expected speedup**: 2-3x over PyTorch CPU inference


### Strategy 4: OpenVINO (Intel-Specific - Best Performance)

**When to use**: Intel CPUs (Xeon, Core), want absolute best CPU performance

**Advantages**:
- Intel-specific optimizations (AVX, AVX-512, VNNI)
- Best-in-class CPU inference performance
- Integrated optimization tools

```python
# Convert PyTorch to OpenVINO IR
# First: Export to ONNX (as above)
# Then: Use Model Optimizer

# Command-line conversion
!mo --input_model model.onnx --output_dir openvino_model --data_type FP16

# Python API
from openvino.runtime import Core

# Load model
ie = Core()
model = ie.read_model(model="openvino_model/model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Inference
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = compiled_model([input_tensor])[0]
```

**Expected speedup**: 3-4x over PyTorch CPU inference on Intel CPUs


### Strategy 5: Batch Size for CPU

**Different from GPU**: Smaller batches often better for CPU

**Why**:
- CPU has smaller cache than GPU memory
- Large batches may not fit in cache → cache misses → slower
- Diminishing returns from batching on CPU

**Recommendation**:
- Start with batch size 1-4
- Profile to find optimal
- Don't assume large batches are better (unlike GPU)

```python
# CPU batch size tuning
def find_optimal_cpu_batch(model, input_shape, max_batch=32):
    model = model.eval()
    results = []

    for batch_size in [1, 2, 4, 8, 16, 32]:
        if batch_size > max_batch:
            break

        test_input = torch.randn(batch_size, *input_shape)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)

        # Benchmark
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(test_input)
        elapsed = time.time() - start

        throughput = (batch_size * 100) / elapsed
        latency = (elapsed / 100) * 1000  # ms

        results.append({
            'batch_size': batch_size,
            'throughput': throughput,
            'latency_ms': latency,
        })

        print(f"Batch {batch_size}: {throughput:.1f} samples/sec, {latency:.2f}ms latency")

    return results
```


## Edge/ARM Optimization

### Strategy 1: INT8 Quantization (2-4x Speedup on ARM)

**When to use**: ARM CPU deployment (Raspberry Pi, mobile, edge devices)

**Why INT8 on ARM**:
- ARM NEON instructions accelerate INT8 operations
- 2-4x faster than FP32 on ARM CPUs
- 4x smaller model size (critical for edge devices)

```python
import torch
from torch.quantization import quantize_dynamic, quantize_static, get_default_qconfig

# Dynamic quantization (easiest, no calibration)
model = load_model().eval()
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # Quantize these layers
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_int8.pth')

# Inference (same API)
with torch.no_grad():
    output = quantized_model(input_tensor)
```

**For better accuracy**: Use static quantization with calibration (see `quantization-for-inference` skill)


### Strategy 2: TensorFlow Lite (Best for ARM/Mobile)

**When to use**:
- ARM edge devices (Raspberry Pi, Coral, mobile)
- Need maximum ARM performance
- Can convert model to TensorFlow Lite

**Advantages**:
- XNNPACK backend (ARM NEON optimizations)
- Highly optimized for edge devices
- Delegate support (GPU, NPU on mobile)

```python
import torch
import tensorflow as tf

# Convert PyTorch to ONNX to TensorFlow to TFLite
# Step 1: PyTorch → ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Step 2: ONNX → TensorFlow (use onnx-tf)
from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

# Step 3: TensorFlow → TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Provide representative dataset for calibration
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 3, 224, 224).astype(np.float32)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Inference with TFLite**:

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="model.tflite",
    num_threads=4  # Use all 4 cores on Raspberry Pi
)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

**Expected speedup**: 3-5x over PyTorch on Raspberry Pi


### Strategy 3: ONNX Runtime for ARM

**When to use**: ARM Linux (Raspberry Pi, Jetson Nano), simpler than TFLite

```python
import onnxruntime as ort

# Export to ONNX (as above)
torch.onnx.export(model, dummy_input, "model.onnx")

# Inference session with ARM optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # Raspberry Pi has 4 cores
sess_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=['CPUExecutionProvider']
)

# Inference
output = session.run(None, {'input': input_data.numpy()})[0]
```

**Quantize ONNX for ARM**:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_int8.onnx",
    weight_type=QuantType.QInt8
)
```


### Strategy 4: Model Architecture for Edge

**When to use**: Inference too slow even after quantization

**Consider smaller architectures**:
- MobileNetV3-Small instead of MobileNetV2
- EfficientNet-Lite instead of EfficientNet
- TinyBERT instead of BERT

**Trade-off**: Accuracy vs speed. Profile to find acceptable balance.

```python
# Compare architectures on edge device
architectures = [
    ('MobileNetV2', models.mobilenet_v2(pretrained=True)),
    ('MobileNetV3-Small', models.mobilenet_v3_small(pretrained=True)),
    ('EfficientNet-B0', models.efficientnet_b0(pretrained=True)),
]

for name, model in architectures:
    model = model.eval()

    # Quantize
    quantized_model = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    # Benchmark
    input_tensor = torch.randn(1, 3, 224, 224)
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = quantized_model(input_tensor)
    elapsed = time.time() - start

    print(f"{name}: {elapsed/100*1000:.2f}ms per inference")
```


## Hardware-Specific Decision Tree

### GPU (NVIDIA)

```
1. Profile with PyTorch profiler or nvidia-smi
   ↓
2. Is GPU utilization low (<50%)?
   YES → Problem:
        - Batch size too small → Increase batch size
        - CPU preprocessing bottleneck → Move preprocessing to GPU or parallelize
        - CPU-GPU transfers → Minimize .cuda()/.cpu() calls
   NO → GPU is bottleneck, optimize model
   ↓
3. Apply optimizations in order:
   a. Increase batch size (measure latency/throughput trade-off)
   b. Mixed precision FP16 (easy 2x speedup if Tensor Cores available)
   c. torch.compile() (easy 1.5-2x speedup, PyTorch 2.0+)
   d. TensorRT (2-5x speedup, more effort)
   e. CUDA graphs (20-30% speedup for small models)
   ↓
4. Measure after each optimization
   ↓
5. If still not meeting requirements:
   - Consider quantization (INT8) → see quantization-for-inference skill
   - Consider model compression → see model-compression-techniques skill
   - Scale horizontally → add more GPU instances
```

### CPU (Intel/AMD)

```
1. Profile with PyTorch profiler or perf
   ↓
2. Check threading configuration
   - torch.get_num_threads() == num physical cores?
   - If not, set torch.set_num_threads(num_cores)
   ↓
3. Apply optimizations in order:
   a. Set intra-op threads to num physical cores
   b. Enable MKLDNN (Intel CPUs)
   c. Use channels-last memory format
   d. Try ONNX Runtime with graph optimizations
   e. If Intel CPU: Try OpenVINO (best performance)
   ↓
4. Measure batch size trade-off (smaller may be better for CPU)
   ↓
5. If still not meeting requirements:
   - Quantize to INT8 → 2-3x speedup on CPU
   - Consider model compression
   - Scale horizontally
```

### Edge/ARM

```
1. Profile on target device (Raspberry Pi, etc.)
   ↓
2. Is inference >100ms per sample?
   YES → Model too large for device
        - Try smaller architecture (MobileNetV3-Small, EfficientNet-Lite)
        - If accuracy allows, use smaller model
   NO → Optimize current model
   ↓
3. Apply optimizations in order:
   a. Quantize to INT8 (2-4x speedup on ARM, critical!)
   b. Set num_threads to device's CPU cores
   c. Convert to TensorFlow Lite with XNNPACK (best ARM performance)
      OR use ONNX Runtime with INT8
   ↓
4. Measure memory usage
   - Model fits in RAM?
   - If not, must use smaller model or offload to storage
   ↓
5. If still not meeting requirements:
   - Use smaller model architecture
   - Consider model pruning
   - Hardware accelerator (Coral TPU, Jetson GPU)
```


## Common Patterns

### Pattern 1: Latency-Critical Online Serving

**Requirements**: <50ms latency, moderate throughput (100-500 req/s)

**Strategy**:
```python
# 1. Small batch size for low latency
batch_size = 1  # or dynamic batching in serving framework

# 2. Use torch.compile() or TensorRT
model = torch.compile(model, mode="reduce-overhead")

# 3. FP16 for speed (if accuracy allows)
model = model.half()

# 4. Profile to ensure <50ms
# 5. If CPU: ensure threading configured correctly
```


### Pattern 2: Throughput-Critical Batch Serving

**Requirements**: High throughput (>1000 samples/sec), latency flexible (100-500ms OK)

**Strategy**:
```python
# 1. Large batch size for throughput
batch_size = 64  # or maximum that fits in memory

# 2. Use TensorRT for maximum optimization
trt_model = torch_tensorrt.compile(model, inputs=[...], enabled_precisions={torch.float16})

# 3. FP16 or INT8 for speed
# 4. Profile to maximize throughput
# 5. Consider CUDA graphs for fixed-size batches
```


### Pattern 3: Edge Deployment (Raspberry Pi)

**Requirements**: <500ms latency, limited memory (1-2GB), ARM CPU

**Strategy**:
```python
# 1. Quantize to INT8 (critical for ARM)
quantized_model = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

# 2. Convert to TensorFlow Lite with XNNPACK
# (see TFLite section above)

# 3. Set threads to device cores (4 for Raspberry Pi 4)
# 4. Profile on device (not on development machine!)
# 5. If too slow, use smaller architecture (MobileNetV3-Small)
```


### Pattern 4: Multi-GPU Inference

**Requirements**: Very high throughput, multiple GPUs available

**Strategy**:
```python
# Option 1: DataParallel (simple, less efficient)
model = torch.nn.DataParallel(model)

# Option 2: Pipeline parallelism (large models)
# Split model across GPUs
model.layer1.to('cuda:0')
model.layer2.to('cuda:1')

# Option 3: Model replication with load balancer (best throughput)
# Run separate inference server per GPU
# Use NGINX or serving framework to distribute requests
```


## Memory vs Compute Trade-offs

### Memory-Constrained Scenarios

**Symptoms**: OOM errors, model barely fits in memory

**Optimizations** (trade compute for memory):
1. **Reduce precision**: FP16 (2x memory reduction) or INT8 (4x reduction)
2. **Reduce batch size**: Smaller batches use less memory
3. **Gradient checkpointing**: (Training only) Recompute activations during backward
4. **Model pruning**: Remove unnecessary parameters
5. **Offload to CPU**: Store some layers/activations on CPU, transfer to GPU when needed

```python
# Example: Reduce precision
model = model.half()  # FP32 → FP16 (2x memory reduction)

# Example: Offload to CPU
class OffloadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.layer1.to('cuda')
        self.model.layer2.to('cpu')  # Offload to CPU
        self.model.layer3.to('cuda')

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.layer2(x.cpu()).cuda()  # Transfer CPU → GPU
        x = self.model.layer3(x)
        return x
```


### Compute-Constrained Scenarios

**Symptoms**: Low throughput, long latency, GPU/CPU underutilized

**Optimizations** (trade memory for compute):
1. **Increase batch size**: Use available memory for larger batches (higher throughput)
2. **Operator fusion**: Combine operations (TensorRT, torch.compile())
3. **Precision increase**: If accuracy suffers from FP16/INT8, use FP32 (slower but accurate)
4. **Larger model**: If accuracy requirements not met, use larger (slower) model

```python
# Example: Increase batch size
# Find maximum batch size that fits in memory
optimal_batch_size = find_optimal_batch_size(model, input_shape)

# Example: Operator fusion (TensorRT)
trt_model = torch_tensorrt.compile(model, inputs=[...], enabled_precisions={torch.float16})
```


## Profiling Checklist

Before optimizing, profile to answer:

### GPU Profiling Questions
- [ ] What is GPU utilization? (nvidia-smi)
- [ ] What is memory utilization?
- [ ] What are the slowest operations? (PyTorch profiler)
- [ ] Is there CPU-GPU transfer overhead? (.cuda()/.cpu() calls)
- [ ] Is batch size optimal? (measure latency/throughput)
- [ ] Are Tensor Cores being used? (FP16/INT8 operations)

### CPU Profiling Questions
- [ ] What is CPU utilization? (all cores used?)
- [ ] What is threading configuration? (torch.get_num_threads())
- [ ] What are the slowest operations? (PyTorch profiler)
- [ ] Is MKLDNN enabled? (Intel CPUs)
- [ ] Is batch size optimal? (may be smaller for CPU)

### Edge Profiling Questions
- [ ] What is inference latency on target device? (not development machine!)
- [ ] What is memory usage? (fits in device RAM?)
- [ ] Is model quantized to INT8? (critical for ARM)
- [ ] Is threading configured for device cores?
- [ ] Is model architecture appropriate for device? (too large?)


## Common Pitfalls

### Pitfall 1: Optimizing Without Profiling

**Mistake**: Applying optimizations blindly without measuring bottleneck

**Example**:
```python
# Wrong: Apply TensorRT without profiling
trt_model = torch_tensorrt.compile(model, ...)

# Right: Profile first
with torch.profiler.profile() as prof:
    output = model(input)
print(prof.key_averages().table())
# Then optimize based on findings
```

**Why wrong**: May optimize wrong part of pipeline (e.g., model is fast, preprocessing is slow)


### Pitfall 2: GPU Optimization for CPU Deployment

**Mistake**: Using GPU-specific optimizations for CPU deployment

**Example**:
```python
# Wrong: TensorRT for CPU deployment
trt_model = torch_tensorrt.compile(model, ...)  # TensorRT requires NVIDIA GPU!

# Right: Use CPU-optimized framework
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
```


### Pitfall 3: Ignoring torch.cuda.synchronize() in GPU Timing

**Mistake**: Measuring GPU time without synchronization (measures kernel launch, not execution)

**Example**:
```python
# Wrong: Inaccurate timing
start = time.time()
output = model(input.cuda())
elapsed = time.time() - start  # Only measures kernel launch!

# Right: Synchronize before measuring
torch.cuda.synchronize()
start = time.time()
output = model(input.cuda())
torch.cuda.synchronize()  # Wait for GPU to finish
elapsed = time.time() - start  # Accurate GPU execution time
```


### Pitfall 4: Batch Size "Bigger is Better"

**Mistake**: Using largest possible batch size without considering latency

**Example**:
```python
# Wrong: Maximum batch size without measuring latency
batch_size = 256  # May violate latency SLA!

# Right: Measure latency vs throughput trade-off
benchmark_batch_sizes(model, input_shape, batch_sizes=[1, 4, 8, 16, 32, 64])
# Select batch size that meets latency requirement
```

**Why wrong**: Large batches increase latency (queue time + compute time), may violate SLA


### Pitfall 5: Not Validating Accuracy After Optimization

**Mistake**: Deploying FP16/INT8 model without checking accuracy

**Example**:
```python
# Wrong: Deploy quantized model without validation
quantized_model = quantize_dynamic(model, ...)
# Deploy immediately

# Right: Validate accuracy first
validate_fp16_accuracy(model, test_loader, tolerance=0.01)
if validation_passes:
    deploy(quantized_model)
```


### Pitfall 6: Over-Optimizing When Requirements Already Met

**Mistake**: Spending effort optimizing when already meeting requirements

**Example**:
```python
# Current: 20ms latency, requirement is <50ms
# Wrong: Spend days optimizing to 10ms (unnecessary)

# Right: Check if requirements met
if current_latency < required_latency:
    print("Requirements met, skip optimization")
```


### Pitfall 7: Wrong Threading Configuration (CPU)

**Mistake**: Not setting intra-op threads, or oversubscribing cores

**Example**:
```python
# Wrong: Default threading (only uses 4-8 cores on 32-core machine)
# (no torch.set_num_threads() call)

# Wrong: Oversubscription
torch.set_num_threads(32)  # Intra-op threads
torch.set_num_interop_threads(32)  # Inter-op threads (total 64 threads on 32 cores!)

# Right: Set intra-op to num cores, disable inter-op
torch.set_num_threads(32)
torch.set_num_interop_threads(1)
```


## When NOT to Optimize

**Skip hardware optimization when**:
1. **Requirements already met**: Current performance satisfies latency/throughput SLA
2. **Model is not bottleneck**: Profiling shows preprocessing or postprocessing is slow
3. **Development phase**: Still iterating on model architecture (optimize after finalizing)
4. **Accuracy degradation**: Optimization (FP16/INT8) causes unacceptable accuracy loss
5. **Rare inference**: Model runs infrequently (e.g., 1x per hour), optimization effort not justified

**Red flag**: Spending days optimizing when requirements already met or infrastructure scaling is cheaper.


## Integration with Other Skills

### With quantization-for-inference
- **This skill**: Hardware-aware quantization deployment (INT8 on CPU vs GPU vs ARM)
- **quantization-for-inference**: Quantization techniques (PTQ, QAT, calibration)
- **Use both**: When quantization is part of hardware optimization strategy

### With model-compression-techniques
- **This skill**: Hardware optimization (batching, frameworks, profiling)
- **model-compression-techniques**: Model size reduction (pruning, distillation)
- **Use both**: When both hardware optimization and model compression needed

### With model-serving-patterns
- **This skill**: Optimize model inference on hardware
- **model-serving-patterns**: Serve optimized model via API/container
- **Sequential**: Optimize model first (this skill), then serve (model-serving-patterns)

### With production-monitoring-and-alerting
- **This skill**: Optimize for target latency/throughput
- **production-monitoring-and-alerting**: Monitor actual latency/throughput in production
- **Feedback loop**: Monitor performance, optimize if degraded


## Success Criteria

You've succeeded when:
- ✅ Profiled before optimizing (identified actual bottleneck)
- ✅ Selected hardware-appropriate optimizations (GPU vs CPU vs edge)
- ✅ Measured performance before/after each optimization
- ✅ Met latency/throughput/memory requirements
- ✅ Validated accuracy after optimization (if using FP16/INT8)
- ✅ Considered cost vs benefit (optimization effort vs infrastructure scaling)
- ✅ Documented optimization choices and trade-offs
- ✅ Avoided premature optimization (requirements already met)


## References

**Profiling**:
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- Intel VTune: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

**GPU Optimization**:
- TensorRT: https://developer.nvidia.com/tensorrt
- torch.compile(): https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- CUDA Graphs: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs

**CPU Optimization**:
- ONNX Runtime: https://onnxruntime.ai/docs/performance/tune-performance.html
- OpenVINO: https://docs.openvino.ai/latest/index.html
- MKLDNN: https://github.com/oneapi-src/oneDNN

**Edge Optimization**:
- TensorFlow Lite: https://www.tensorflow.org/lite/performance/best_practices
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/

**Batch Size Tuning**:
- Dynamic Batching: https://github.com/pytorch/serve/blob/master/docs/batch_inference_with_ts.md
- TorchServe Batching: https://pytorch.org/serve/batch_inference.html
