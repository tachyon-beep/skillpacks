
# Quantization for Inference Skill

## When to Use This Skill

Use this skill when you observe these symptoms:

**Performance Symptoms:**
- Model inference too slow on CPU (e.g., >10ms when need <5ms)
- Batch processing taking too long (low throughput)
- Need to serve more requests per second with same hardware

**Size Symptoms:**
- Model too large for edge devices (e.g., 100MB+ for mobile)
- Want to fit more models in GPU memory
- Memory-constrained deployment environment

**Deployment Symptoms:**
- Deploying to CPU servers (quantization gives 2-4× CPU speedup)
- Deploying to edge devices (mobile, IoT, embedded systems)
- Cost-sensitive deployment (smaller models = lower hosting costs)

**When NOT to use this skill:**
- Model already fast enough and small enough (no problem to solve)
- Deploying exclusively on GPU with no memory constraints (modest benefit)
- Prototyping phase where optimization is premature
- Model so small that quantization overhead not worth it (e.g., <5MB)

**For LLM-specific generation-quality tradeoffs (perplexity vs. format choice, KV-cache quantization, speculative decoding):** see `yzmir-llm-specialist/llm-inference-optimization.md`. This sheet owns the quantization mechanics and cross-modality coverage; the LLM specialist sheet owns generation-quality decisions.

## Core Principle

**Quantization trades precision for performance.**

Quantization converts high-precision numbers (FP32: 32 bits) to low-precision integers (INT8: 8 bits or INT4: 4 bits). This provides:
- **4-8× smaller model size** (fewer bits per parameter)
- **2-4× faster inference on CPU** (INT8 operations faster than FP32)
- **Small accuracy loss** (typically 0.5-1% for INT8)

**Formula:** Lower precision (FP32 → INT8 → INT4) = Smaller size + Faster inference + More accuracy loss

The skill is choosing the **right precision for your accuracy tolerance**.

## PyTorch namespace note

PyTorch 2.x has consolidated quantization under `torch.ao.quantization`. The legacy `torch.quantization` module forwards to `torch.ao.quantization` and is deprecated for new code. There are two recommended paths:

- **Eager-mode quantization** under `torch.ao.quantization` — the dynamic / static / QAT flow shown in Parts 1–3 below. This is the path you use for models that already work in eager mode and that you want to quantize without a graph capture.
- **PT2E (PyTorch 2 Export) quantization** — the graph-based path using `torch.export.export()` + `prepare_pt2e()` + `convert_pt2e()` from `torch.ao.quantization.quantize_pt2e`. This is the recommended path for new work that targets `torch.compile` / Inductor or specific hardware backends (X86Inductor, XNNPACK, executorch). See Part 5b. ([PyTorch quantization docs](https://docs.pytorch.org/docs/main/quantization.html), [PT2E tutorial](https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html))

Throughout this sheet imports use `torch.ao.quantization`. If you see `torch.quantization` in older code it still works as an alias, but migrate it.

## Quantization Framework

```
┌────────────────────────────────────────────┐
│   1. Recognize Quantization Need           │
│   CPU/Edge + (Slow OR Large)               │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   2. Choose Quantization Type              │
│   Dynamic → Static → QAT (increasing cost) │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   3. Calibrate (if Static/QAT)             │
│   100-1000 representative samples          │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   4. Validate Accuracy Trade-offs          │
│   Baseline vs Quantized accuracy           │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   5. Decide: Accept or Iterate             │
│   <2% loss → Deploy                        │
│   >2% loss → Try QAT or different precision│
└────────────────────────────────────────────┘
```

## Part 1: Quantization Types

### Type 1: Dynamic Quantization

**What it does:** Quantizes weights to INT8, keeps activations in FP32.

**When to use:**
- Simplest quantization (no calibration needed)
- Primary goal is size reduction
- Batch processing where latency less critical
- Quick experiment to see if quantization helps

**Benefits:**
- 4× size reduction (weights are 75% of model size)
- 1.2-1.5× CPU speedup (modest, because activations still FP32)
- Minimal accuracy loss (~0.2-0.5%)
- No calibration data needed

**Limitations:**
- Limited CPU speedup (activations still FP32)
- Not optimal for edge devices needing maximum performance

**PyTorch implementation:**

```python
import torch
import torch.ao.quantization as tq

# WHY: Dynamic quantization is simplest - just one function call
# No calibration data needed because activations stay FP32
model = torch.load('model.pth')
model.eval()  # WHY: Must be in eval mode (no batchnorm updates)

# WHY: Specify which layers to quantize (Linear, LSTM, etc.)
# These layers benefit most from quantization
quantized_model = tq.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},  # WHY: Quantize Linear layers only
    dtype=torch.qint8                # WHY: INT8 is standard precision
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized_dynamic.pth')

# Verify size reduction
import os
original_size = os.path.getsize('model.pth') / (1024 ** 2)  # MB
quantized_size = os.path.getsize('model_quantized_dynamic.pth') / (1024 ** 2)
print(f"Original: {original_size:.1f}MB → Quantized: {quantized_size:.1f}MB")
print(f"Size reduction: {original_size / quantized_size:.1f}×")
```

**Example use case:** BERT classification model where primary goal is reducing size from 440MB to 110MB for easier deployment.


### Type 2: Static Quantization (Post-Training Quantization)

**What it does:** Quantizes both weights and activations to INT8.

**When to use:**
- Need maximum CPU speedup (2-4×)
- Deploying to CPU servers or edge devices
- Can afford calibration step (5-10 minutes)
- Primary goal is inference speed

**Benefits:**
- 4× size reduction (same as dynamic)
- 2-4× CPU speedup (both weights and activations INT8)
- No retraining required (post-training)
- Acceptable accuracy loss (~0.5-1%)

**Requirements:**
- Needs calibration data (100-1000 samples from validation set)
- Slightly more complex setup than dynamic

**PyTorch implementation (eager-mode static quantization):**

```python
import torch
import torch.ao.quantization as tq

def calibrate_model(model, calibration_loader):
    """
    Calibrate model by running representative data through it.

    WHY: Static quantization needs to know activation ranges.
    Calibration finds min/max values for each activation layer.
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            model(data)
            if batch_idx >= 100:  # WHY: 100 batches usually sufficient
                break
    return model

# Step 1: Prepare model for quantization
model = torch.load('model.pth')
model.eval()

# WHY: Insert quantization/dequantization stubs at boundaries
# This tells PyTorch where to convert between FP32 and INT8
model.qconfig = tq.get_default_qconfig('x86')  # WHY: 'x86' is the modern
                                               # backend alias (replaces
                                               # 'fbgemm' from PyTorch 2.x)
tq.prepare(model, inplace=True)

# Step 2: Calibrate with representative data
# WHY: Must use data from training/validation set, not random data
# Calibration finds activation ranges - needs real distribution
calibration_dataset = torch.utils.data.Subset(
    val_dataset,
    indices=range(1000)  # WHY: 1000 samples sufficient for most models
)
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=False  # WHY: Order doesn't matter for calibration
)

model = calibrate_model(model, calibration_loader)

# Step 3: Convert to quantized model
tq.convert(model, inplace=True)

# Save quantized model
torch.save(model.state_dict(), 'model_quantized_static.pth')

# Benchmark speed improvement
import time

def benchmark(model, data, num_iterations=100):
    """WHY: Warm up model first, then measure average latency."""
    model.eval()
    for _ in range(10):  # warm up
        model(data)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(data)
    end = time.time()
    return (end - start) / num_iterations * 1000  # ms per inference

test_data = torch.randn(1, 3, 224, 224)
baseline_latency = benchmark(original_model, test_data)
quantized_latency = benchmark(model, test_data)
print(f"Baseline: {baseline_latency:.2f}ms")
print(f"Quantized: {quantized_latency:.2f}ms")
print(f"Speedup: {baseline_latency / quantized_latency:.2f}×")
```

**Example use case:** ResNet50 image classifier for CPU inference - need <5ms latency, achieve 4ms with static quantization (vs 15ms baseline).


### Type 3: Quantization-Aware Training (QAT)

**What it does:** Simulates quantization during training to minimize accuracy loss.

**When to use:**
- Static quantization accuracy loss too large (>2%)
- Need best possible accuracy with INT8
- Can afford retraining (hours to days)
- Critical production system with strict accuracy requirements

**Benefits:**
- Best accuracy (~0.1-0.3% loss vs 0.5-1% for static)
- 4× size reduction (same as dynamic/static)
- 2-4× CPU speedup (same as static)

**Limitations:**
- Requires retraining (most expensive option)
- Takes hours to days depending on model size
- More complex implementation

**PyTorch implementation:**

```python
import torch
import torch.ao.quantization as tq

def train_one_epoch_qat(model, train_loader, optimizer, criterion):
    """
    Train one epoch with quantization-aware training.

    WHY: QAT inserts fake quantization ops during training.
    Model learns to be robust to quantization errors.
    """
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model

# Step 1: Prepare model for QAT
model = torch.load('model.pth')
model.train()

# WHY: QAT config includes fake quantization ops
# These simulate quantization during forward pass
model.qconfig = tq.get_default_qat_qconfig('x86')
tq.prepare_qat(model, inplace=True)

# Step 2: Train with quantization-aware training
# WHY: Model learns to compensate for quantization errors
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # low LR for fine-tuning
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5  # WHY: Usually 5-10 epochs sufficient for QAT fine-tuning
for epoch in range(num_epochs):
    model = train_one_epoch_qat(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} complete")

# Step 3: Convert to quantized model
model.eval()
tq.convert(model, inplace=True)

torch.save(model.state_dict(), 'model_quantized_qat.pth')
```

**Example use case:** Medical imaging model where accuracy is critical - static quantization gives 2% accuracy loss, QAT reduces to 0.3%.


## Part 2: Quantization Type Decision Matrix

| Type | Complexity | Calibration | Retraining | Size Reduction | CPU Speedup | Accuracy Loss |
|------|-----------|-------------|------------|----------------|-------------|---------------|
| **Dynamic** | Low | No | No | 4× | 1.2-1.5× | ~0.2-0.5% |
| **Static (eager)** | Medium | Yes | No | 4× | 2-4× | ~0.5-1% |
| **Static (PT2E export)** | Medium | Yes | No | 4× | 2-4× (Inductor backend) | ~0.5-1% |
| **QAT** | High | Yes | Yes | 4× | 2-4× | ~0.1-0.3% |

**Decision flow:**
1. Start with **dynamic quantization**: Simplest, verify quantization helps
2. Upgrade to **static quantization**: If need more speedup, can afford calibration. Use eager-mode for legacy models, PT2E for new work targeting Inductor / specific backends.
3. Use **QAT**: Only if accuracy loss from static too large (rare)

**Why this order?** Incremental cost. Dynamic is free (5 minutes), static is cheap (15 minutes), QAT is expensive (hours/days). Don't pay for QAT unless you need it.


## Part 3: Calibration Best Practices

### What is Calibration?

**Purpose:** Find min/max ranges for each activation layer.

**Why needed:** Static quantization needs to know activation ranges to map FP32 → INT8. Without calibration, ranges are wrong → accuracy collapses.

**How it works:**
1. Run representative data through model
2. Record min/max activation values per layer
3. Use these ranges to quantize activations at inference time

### Calibration Data Requirements

**Data source:**
- Use validation set samples (matches training distribution)
- Don't use random images from internet (different distribution)
- Don't use single image repeated (insufficient coverage)
- Don't use training set that doesn't match deployment (distribution shift)

**Data size:**
- **Minimum:** 100 samples (sufficient for simple models)
- **Recommended:** 500-1000 samples (better coverage)
- **Maximum:** Full validation set is overkill (slow, no benefit)

**Data characteristics:**
- Must cover range of inputs model sees in production
- Include edge cases (bright/dark images, long/short text)
- Distribution should match deployment, not just training
- Class balance less important than input diversity

**Example calibration data selection:**

```python
import torch
import numpy as np

def select_calibration_data(val_dataset, num_samples=1000):
    """
    Select diverse calibration samples from validation set.

    WHY: Want samples that cover range of activation values.
    Random selection from validation set usually sufficient.
    """
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    return torch.utils.data.Subset(val_dataset, indices)

calibration_dataset = select_calibration_data(val_dataset, num_samples=1000)
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=False  # WHY: Order doesn't matter for calibration
)
```

### Common Calibration Pitfalls

**Pitfall 1: Using wrong data distribution**
- Wrong: "Random images from internet" for ImageNet-trained model
- Right: Use ImageNet validation set samples

**Pitfall 2: Too few samples**
- Wrong: 10 samples (insufficient coverage of activation ranges)
- Right: 100-1000 samples (good coverage)

**Pitfall 3: Using training data that doesn't match deployment**
- Wrong: Calibrate on sunny outdoor images, deploy on indoor images
- Right: Calibrate on data matching deployment distribution

**Pitfall 4: Skipping calibration validation**
- Wrong: Calibrate once, assume it works
- Right: Validate accuracy after calibration to verify ranges are good


## Part 4: Precision Selection (INT8 vs INT4 vs FP16 vs FP8)

### Precision Spectrum

| Precision | Bits | Size vs FP32 | Speedup (CPU) | Typical Accuracy Loss |
|-----------|------|--------------|---------------|----------------------|
| **FP32** | 32 | 1× | 1× | 0% (baseline) |
| **BF16/FP16** | 16 | 2× | 1.5× | <0.1% |
| **FP8 (E4M3/E5M2)** | 8 | 4× | 2-3× (on Hopper/Ada/Blackwell) | ~0.1-0.5% |
| **INT8** | 8 | 4× | 2-4× | 0.5-1% |
| **MXFP6** | 6 + scale | ~5× | hardware-dependent (Blackwell native) | task-dependent |
| **INT4 / MXFP4 / NF4** | 4 + scale | 8× | 4-8× | 1-3% |
| **2-3 bit (AQLM, etc.)** | 2-3 + scale | 10-16× | hardware-limited | 2-5% |

**Trade-off:** Lower precision = Smaller size + Faster inference + More accuracy loss

### When to Use Each Precision

**BF16 / FP16 (Half Precision):**
- GPU inference (Tensor Cores optimized for FP16/BF16)
- Need minimal accuracy loss (<0.1%)
- Default datatype for modern LLM training and inference baseline
- BF16 preferred for training stability; FP16 fine for inference of already-trained weights

**FP8 (E4M3 for forward, E5M2 for backward / activations):**
- Hopper (H100), Ada (RTX 40-series), Blackwell (B100/B200) GPUs natively support FP8 in Tensor Cores via NVIDIA Transformer Engine
- E4M3 = 4 exponent bits, 3 mantissa bits — better precision, used for weights and forward activations
- E5M2 = 5 exponent bits, 2 mantissa bits — wider range, used for gradients
- Best balance of accuracy and speed when hardware supports it
- See [NVIDIA Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/) for E4M3/E5M2 conventions

**INT8 (Standard Quantization):**
- CPU inference (INT8 operations fast on CPU, AVX-VNNI / ARM dot-product)
- Edge device deployment
- Good balance of size/speed/accuracy
- **Most common choice** for non-LLM production deployment
- Example: image classification on mobile devices

**INT4 / NF4 / MXFP4 (Aggressive Quantization):**
- LLM weight-only quantization is the dominant use case
- INT4 or NF4 (QLoRA) for weight-only LLM compression — see Part 7
- MXFP4 (OCP MX format) on Blackwell for native 4-bit Tensor Core throughput
- Use for non-LLM models only when memory pressure is extreme

**MXFP6 (block-scaled 6-bit):**
- Blackwell-native, but no raw throughput advantage over MXFP8 — use when 6-bit gives accuracy MXFP4 cannot deliver and MXFP8 is too large

**Sub-4-bit (AQLM, HQQ 2-bit):**
- Only for LLMs where 4-bit is still too large (e.g., 70B+ on consumer GPUs)
- Accept 2-5% quality loss; verify on your eval set


## Part 5: ONNX Quantization (Cross-Framework)

**When to use:** Deploying to ONNX Runtime (CPU/edge devices) or need cross-framework compatibility.

### ONNX Static Quantization

```python
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat
import numpy as np

class CalibrationDataReaderWrapper(CalibrationDataReader):
    """
    WHY: ONNX requires custom calibration data reader.
    This class feeds calibration data to ONNX quantization engine.
    """
    def __init__(self, calibration_data):
        self.calibration_data = calibration_data
        self.iterator = iter(calibration_data)

    def get_next(self):
        """Called by ONNX to get next calibration batch."""
        try:
            data, _ = next(self.iterator)
            return {"input": data.numpy()}
        except StopIteration:
            return None

# Step 1: Export PyTorch model to ONNX
model = torch.load('model.pth')
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=17,  # WHY: Opset 17 is the modern stable target
                       # (covers all common quantization ops + LayerNorm)
)

# Step 2: Prepare calibration data
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=1,
    shuffle=False
)
calibration_reader = CalibrationDataReaderWrapper(calibration_loader)

# Step 3: Quantize ONNX model
quantize_static(
    'model.onnx',
    'model_quantized.onnx',
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ  # WHY: QDQ format is the modern default,
                                  # compatible with most backends
)
```

**ONNX advantages:**
- Cross-framework (works with PyTorch, TensorFlow, etc.)
- Optimized ONNX Runtime for CPU inference
- Good hardware backend support (x86, ARM, CoreML, NNAPI)


## Part 5b: PT2E Export Quantization (PyTorch 2 graph path)

**When to use:** New PyTorch quantization work, especially when targeting `torch.compile` / Inductor, X86 server inference, ARM XNNPACK, or ExecuTorch on-device.

PT2E is the graph-based replacement for FX-graph quantization. The flow is **export → prepare → calibrate → convert**, parameterized by a `Quantizer` object that encodes the target backend's constraints.

```python
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

# Step 1: Capture model graph with torch.export
# WHY: torch.export traces the model into an ExportedProgram - a stable
# graph IR that downstream backends consume. This replaces the older
# FX symbolic_trace path.
model = torch.load('model.pth')
model.eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

exported_program = torch.export.export(model, example_inputs)
graph_module = exported_program.module()

# Step 2: Pick a Quantizer for your target backend
# WHY: Quantizer encodes the backend's op-support and dtype constraints.
# X86InductorQuantizer targets server CPU through torch.compile + Inductor.
# Other quantizers exist for XNNPACK (ARM), ExecuTorch backends, etc.
quantizer = X86InductorQuantizer()
quantizer.set_global(get_default_x86_inductor_quantization_config())

# Step 3: Prepare - inserts observers
prepared_model = prepare_pt2e(graph_module, quantizer)

# Step 4: Calibrate
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(calibration_loader):
        prepared_model(data)
        if batch_idx >= 100:
            break

# Step 5: Convert - swaps observers for quantized ops
quantized_model = convert_pt2e(prepared_model)

# Step 6: Lower with torch.compile for execution
# WHY: PT2E quantized graphs are designed to be consumed by Inductor
# for fused INT8 codegen on CPU.
optimized_model = torch.compile(quantized_model)
```

References: [PyTorch quantization overview](https://docs.pytorch.org/docs/main/quantization.html), [PT2E quantization index](https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html), [PT2E with X86 backend tutorial](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html), [`torch.ao.quantization.quantize_pt2e` source](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantize_pt2e.py).

**Eager vs PT2E summary:**

| Aspect | Eager (`torch.ao.quantization`) | PT2E (`quantize_pt2e`) |
|--------|---------------------------------|------------------------|
| Graph capture | None (module-level) | `torch.export.export()` |
| Best for | Legacy code, custom modules | New work, Inductor backends, on-device export |
| Backends | FBGEMM, QNNPACK, x86 | X86Inductor, XNNPACK, ExecuTorch backends |
| QAT | `prepare_qat` | `prepare_qat_pt2e` |


## Part 6: Accuracy Validation (Critical Step)

### Why Accuracy Validation Matters

Quantization is **lossy compression**. Must measure accuracy impact:
- Some models tolerate quantization well (<0.5% loss)
- Some models sensitive to quantization (>2% loss)
- Some layers more sensitive than others
- Can't assume quantization is safe without measuring

### Validation Methodology

```python
def validate_quantization(original_model, quantized_model, val_loader):
    """
    Validate quantization by comparing accuracy.

    WHY: Quantization is lossy - must measure impact.
    Compare baseline vs quantized on same validation set.
    """
    def evaluate(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total

    baseline_acc = evaluate(original_model, val_loader)
    quantized_acc = evaluate(quantized_model, val_loader)
    accuracy_loss = baseline_acc - quantized_acc

    return {
        'baseline_acc': baseline_acc,
        'quantized_acc': quantized_acc,
        'accuracy_loss': accuracy_loss,
        'acceptable': accuracy_loss < 2.0  # <2% loss usually acceptable
    }

results = validate_quantization(original_model, quantized_model, val_loader)
print(f"Baseline accuracy: {results['baseline_acc']:.2f}%")
print(f"Quantized accuracy: {results['quantized_acc']:.2f}%")
print(f"Accuracy loss: {results['accuracy_loss']:.2f}%")
print(f"Acceptable: {results['acceptable']}")
```

### Acceptable Accuracy Thresholds

**General guidelines:**
- **<1% loss:** Excellent quantization result
- **1-2% loss:** Acceptable for most applications
- **2-3% loss:** Consider QAT to reduce loss
- **>3% loss:** Quantization may not be suitable for this model

**Task-specific thresholds:**
- Image classification: 1-2% top-1 accuracy loss acceptable
- Object detection: 1-2% mAP loss acceptable
- NLP classification: 0.5-1% accuracy loss acceptable
- LLM generation: <0.1 perplexity increase, plus task-specific eval (see llm-specialist)
- Medical/safety-critical: <0.5% loss required (use QAT)


## Part 7: LLM Quantization Reference Matrix

LLMs need different quantization techniques than vision/NLP-classification models because:
- They are dominated by very large `nn.Linear` weights (so weight-only quantization is highly effective).
- Activations have severe outliers in specific channels — naive INT8 collapses quality.
- Inference is memory-bandwidth bound on a single GPU, so smaller weights translate directly into throughput.

This part documents the named techniques. **For "should I use AWQ or GPTQ for *this* model" generation-quality decisions, KV-cache quantization, and speculative-decoding interactions, see `yzmir-llm-specialist/llm-inference-optimization.md`.**

### 7.1 Named techniques

**AWQ — Activation-aware Weight Quantization**
Lin et al., MIT Han Lab, 2023; MLSys 2024 best paper. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978). Observes that ~1% of "salient" weights dominate quality; uses activation magnitudes to find a per-channel scale that protects them, then does weight-only INT4 quantization without backprop. Tool: [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) (community) or [llm-awq](https://github.com/mit-han-lab/llm-awq) (official). Hardware: Ampere+ (uses INT4 packed kernels).

**GPTQ — Generative Pretrained Transformer Quantization**
Frantar et al., 2022. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). Layer-wise Hessian-based weight quantization at 3-4 bits using approximate second-order information from a small calibration set. Tool: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (legacy) and [GPTQModel](https://github.com/ModelCloud/GPTQModel) (active fork). Hardware: any (CUDA, ROCm, CPU kernels exist).

**AQLM — Additive Quantization for Language Models**
Egiazarian et al., 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118). Generalizes additive quantization (multiple learned codebooks summed) to LLMs; achieves Pareto-optimal accuracy below 3 bits-per-parameter, including practical ~2-bit results. Tool: [AQLM repo](https://github.com/Vahe1994/AQLM). Hardware: GPU (custom CUDA kernels).

**HQQ — Half-Quadratic Quantization**
Mobius Labs, 2023. Calibration-free weight quantization solved as a half-quadratic optimization in closed form per row, supporting 1-8 bit. Quantizes very large models in minutes without a calibration dataset. Tool: [hqq](https://github.com/mobiusml/hqq).

**bitsandbytes NF4 / FP4**
Used in QLoRA. NF4 (4-bit NormalFloat) is a non-uniform datatype information-theoretically optimal for normally-distributed weights; FP4 is 4-bit floating-point. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). NF4 outperforms FP4 by ~1pp at the same bit-width. Tool: [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) integrated into HF Transformers.

**SmoothQuant**
Xiao et al., MIT, 2022; ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438). Mathematically equivalent transformation that migrates the "difficulty" of quantizing activation outliers into the weights. Enables W8A8 (INT8 weights + INT8 activations) for LLMs without retraining. Tool: [smoothquant repo](https://github.com/mit-han-lab/smoothquant). Pairs well with INT8 Tensor Cores.

**GGUF + k-quants**
File format and quantization scheme used by [llama.cpp](https://github.com/ggml-org/llama.cpp) and the broader GGML ecosystem. K-quants (`Q2_K`, `Q3_K`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, etc.) are block-wise mixed-precision quantizations tuned for CPU and Apple Silicon inference. `Q4_K_M` and `Q5_K_M` are the practical default trade-off points for most users.

**FP8 (E4M3, E5M2)**
Native Tensor Core support starting on NVIDIA Hopper (H100/H200), Ada (RTX 40-series), and Blackwell (B100/B200). E4M3 = 4 exponent / 3 mantissa bits (higher precision, used for weights and forward activations); E5M2 = 5 exponent / 2 mantissa bits (wider range, used for gradients). Tooling: [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Used by both training (with delayed scaling) and inference (with per-tensor scales).

**MXFP4 / MXFP6 — OCP Microscaling formats**
[OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), September 2023. Block-scaled formats with a 32-element block size and an 8-bit shared exponent (`E8M0`). MXFP4 uses E2M1 elements; MXFP6 supports E2M3 and E3M2. Native Blackwell Tensor Core support: MXFP4 has 2× the raw throughput of MXFP8; MXFP6 matches MXFP8 throughput but provides extra accuracy headroom. Reference paper: ["Microscaling Data Formats for Deep Learning"](https://arxiv.org/abs/2310.10537).

### 7.2 Hardware compatibility matrix

| Format | Ampere (A100, A40) | Ada (RTX 40, L40) | Hopper (H100, H200) | Blackwell (B100/B200) | x86 CPU | ARM CPU | Apple Silicon |
|--------|---|---|---|---|---|---|---|
| INT8 | yes (Tensor Cores) | yes | yes | yes | yes (AVX-VNNI) | yes (dot-product) | yes |
| FP16 / BF16 | yes | yes | yes | yes | bf16 partial | bf16 (SVE) | yes |
| FP8 (E4M3/E5M2) | no | yes (Ada) | yes (native) | yes (native) | no | no | no |
| INT4 weight-only (AWQ/GPTQ kernels) | yes | yes | yes | yes | partial (ggml) | partial | yes (MLX, llama.cpp) |
| NF4 (bitsandbytes) | yes | yes | yes | yes | no | no | no |
| MXFP4 / MXFP6 | no | no | no | yes (native) | no | no | no |
| GGUF k-quants | yes (CUDA) | yes | yes | yes | yes (primary target) | yes | yes (primary target) |
| AQLM (~2-bit) | yes | yes | yes | yes | partial | no | partial |

### 7.3 Tool/format support matrix

| Engine | AWQ | GPTQ | AQLM | HQQ | NF4/FP4 | SmoothQuant W8A8 | GGUF | FP8 | MXFP4/6 |
|--------|-----|------|------|-----|---------|------------------|------|-----|---------|
| **AutoAWQ** | yes (native) | no | no | no | no | no | no | no | no |
| **AutoGPTQ / GPTQModel** | no | yes (native) | no | no | no | no | no | no | no |
| **llama.cpp / ExLlamaV2** | partial (via convert) | yes (ExLlamaV2) | no | no | no | no | yes (llama.cpp primary) | no | no |
| **vLLM** | yes | yes | yes | partial | yes (via bnb) | yes | partial (limited) | yes (Hopper+) | emerging (Blackwell) |
| **TensorRT-LLM** | yes | yes | no | no | no | yes | no | yes (native) | yes (Blackwell) |
| **Hugging Face Transformers + bitsandbytes** | via AWQ pkg | via GPTQ pkg | via AQLM pkg | via hqq pkg | yes (native) | no | no | partial | no |
| **MLC-LLM** | yes | yes | no | no | no | no | no | no | no |
| **NVIDIA Transformer Engine** | no | no | no | no | no | no | no | yes (primary) | yes (Blackwell) |

(This matrix tracks supported formats, not performance; check current release notes for kernel-level details.)

### 7.4 Quick-pick guidance

- **Self-hosted LLM serving on NVIDIA, want the easy path:** vLLM + AWQ INT4 (or GPTQ INT4 if no AWQ checkpoint exists).
- **Need to fit a 70B+ model on a single 80 GB GPU or 13B on 8 GB:** AQLM 2-bit, or AWQ INT3 if available.
- **CPU / laptop / Apple Silicon:** llama.cpp + GGUF (`Q4_K_M` or `Q5_K_M`).
- **Hopper / Blackwell, want maximum throughput on long-context serving:** TensorRT-LLM + FP8 (E4M3 weights + E4M3 activations) or MXFP4 on Blackwell.
- **Need to fine-tune on consumer hardware:** QLoRA (NF4) via bitsandbytes + PEFT.

For the "AWQ vs GPTQ vs FP8 for *this* particular model" decision, see `yzmir-llm-specialist/llm-inference-optimization.md` — the LLM-specialist sheet covers the perplexity / task-eval tradeoffs.


## Part 8: When NOT to Quantize

### Scenario 1: Already Fast Enough

**Example:** MobileNetV2 (14MB, 3ms CPU latency)
- Quantization: 14MB → 4MB, 3ms → 2ms
- Benefit: 10MB saved, 1ms faster
- Cost: Calibration, validation, testing, debugging
- **Decision:** Not worth effort unless specific requirement

**Rule:** If current performance meets requirements, don't optimize.

### Scenario 2: GPU-Only Deployment with No Memory Constraints

**Example:** ResNet50 on a single GPU with plenty of headroom
- Quantization: 1.5-2× GPU speedup (modest)
- FP16/BF16 already fast on Tensor Cores
- No memory pressure
- **Decision:** Focus on other bottlenecks (data loading, I/O, batching)

**Rule:** Quantization is most beneficial for CPU inference, edge, and memory-constrained GPU. For LLMs, even on big GPUs, weight-only quantization usually wins because inference is memory-bandwidth bound.

### Scenario 3: Accuracy-Critical Applications

**Example:** Medical diagnosis model where misdiagnosis has severe consequences
- Quantization introduces accuracy loss (even if small)
- Risk not worth benefit
- **Decision:** Keep FP32/BF16, optimize other parts (batching, caching)

**Rule:** Safety-critical systems should avoid lossy compression unless thoroughly validated end-to-end.

### Scenario 4: Prototyping Phase

Quantization is optimization — premature at prototype stage. Focus on getting model working first.


## Part 9: Quantization Benchmarks (Expected Results)

These are illustrative ranges from published benchmarks; verify on your hardware and model.

### Image Classification (ResNet50, ImageNet)

| Metric | FP32 Baseline | Dynamic INT8 | Static INT8 | QAT INT8 |
|--------|---------------|--------------|-------------|----------|
| Size | ~98MB | ~25MB (4×) | ~25MB (4×) | ~25MB (4×) |
| CPU Latency | ~15ms | ~12ms (1.25×) | ~4ms (3.75×) | ~4ms (3.75×) |
| Top-1 Accuracy | ~76.1% | ~75.9% (-0.2pp) | ~75.3% (-0.8pp) | ~75.9% (-0.2pp) |

### Object Detection (YOLOv5s, COCO)

| Metric | FP32 Baseline | Static INT8 | QAT INT8 |
|--------|---------------|-------------|----------|
| Size | ~14MB | ~4MB (3.5×) | ~4MB (3.5×) |
| CPU Latency | ~45ms | ~15ms (3×) | ~15ms (3×) |
| mAP@0.5 | ~37.4% | ~36.8% (-0.6pp) | ~37.2% (-0.2pp) |

### NLP Classification (BERT-base, GLUE)

| Metric | FP32 Baseline | Dynamic INT8 | Static INT8 |
|--------|---------------|--------------|-------------|
| Size | ~440MB | ~110MB (4×) | ~110MB (4×) |
| CPU Latency | ~35ms | ~28ms (1.25×) | ~12ms (2.9×) |
| Accuracy | ~93.5% | ~93.2% (-0.3pp) | ~92.8% (-0.7pp) |

### LLM Inference (7B class, single GPU)

Numbers cited from the AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) and GPTQ ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) papers, paraphrased; check published numbers and current kernels for your engine.

| Metric | FP16 Baseline | GPTQ INT4 | AWQ INT4 |
|--------|---------------|-----------|----------|
| Weights size | ~13GB | ~3.5GB (3.7×) | ~3.5GB (3.7×) |
| Tokens/sec (single GPU, decode) | 1× | 2-3× | 2.7-3.9× |
| Perplexity (WikiText-2 7B) | baseline | small increase | smaller increase |

For up-to-date LLM serving throughput numbers (vLLM, TensorRT-LLM, SGLang), see `yzmir-llm-specialist/llm-inference-optimization.md` and the engine release notes.


## Part 10: Common Pitfalls and Solutions

### Pitfall 1: Skipping Accuracy Validation

```python
# WRONG: Deploy without validation
quantized_model = quantize(model)
deploy(quantized_model)

# RIGHT: Validate before deployment
quantized_model = quantize(model)
results = validate_accuracy(original_model, quantized_model, val_loader)
if results['acceptable']:
    deploy(quantized_model)
else:
    print("Accuracy loss too large - try QAT")
```

### Pitfall 2: Using Wrong Calibration Data

Calibrate with random/unrepresentative data → activation ranges wrong → accuracy collapses. Use 100-1000 samples from validation/production-shadow set.

### Pitfall 3: Choosing Wrong Quantization Type

Use dynamic when need static speedup → get 1.2× instead of 3×. Match type to requirements.

### Pitfall 4: Quantizing GPU-Only Deployments Unnecessarily

If you have memory headroom and FP16/BF16 meets latency, quantization may not be worth the integration cost. Exception: LLMs, where weight-only INT4 usually pays for itself via memory bandwidth.

### Pitfall 5: Over-Quantizing (INT4 When INT8 Sufficient)

Larger accuracy loss than necessary. Start with INT8, only go to 4-bit if memory or latency forces it.

### Pitfall 6: Using `torch.quantization` (legacy alias) in new code

```python
# OLD: import torch.quantization  -- deprecated alias
# NEW: import torch.ao.quantization as tq
```

The old namespace forwards but is not the documented path. Use `torch.ao.quantization` for eager mode and `torch.ao.quantization.quantize_pt2e` for the export path.

### Pitfall 7: Assuming All Layers Quantize Equally

Some layers (first conv, final classifier, attention output projections) are more sensitive. Use mixed precision — keep sensitive layers in higher precision, quantize the rest aggressively. PT2E quantizers support per-module configuration via `Quantizer.set_module_name()`.


## Part 11: Decision Framework Summary

### Step 1: Recognize Quantization Need

Symptoms: too slow on CPU, too large for edge, deploying to CPU/edge/memory-constrained GPU, hosting cost pressure.

### Step 2: Choose Quantization Type

```
LLM?
├─ YES → Part 7 matrix (AWQ/GPTQ/AQLM/GGUF/FP8/MXFP4 by hardware)
└─ NO  → Vision/NLP-classification path:
   Start with Dynamic
   ├─ Sufficient? → Deploy
   └─ NO → Static (eager or PT2E)
      ├─ Sufficient and accuracy OK? → Deploy
      └─ NO (>2% accuracy loss) → QAT
```

### Step 3: Calibrate (if Static/QAT)

100-1000 samples from validation set matching deployment distribution.

### Step 4: Validate Accuracy

Baseline vs quantized on the same eval set. If task-specific eval is available (LLM benchmarks, downstream task), use it — perplexity alone is not sufficient for LLMs.

### Step 5: Benchmark Performance

Size, latency, throughput on the actual target hardware. Don't trust benchmarks from different GPUs / kernels.


## Part 12: Production Deployment Checklist

**Accuracy validated**
- [ ] Baseline measured on representative eval set
- [ ] Quantized measured on the same eval set
- [ ] Loss within acceptable threshold for the task
- [ ] For LLMs: task-specific eval, not just perplexity (see llm-evaluation-metrics)

**Performance benchmarked**
- [ ] Size reduction measured
- [ ] Latency measured on target hardware (not dev box)
- [ ] Throughput measured under realistic load
- [ ] For LLMs: tokens/sec at relevant batch sizes and context lengths

**Calibration verified** (if static/QAT)
- [ ] Used 100-1000 samples from validation set
- [ ] Distribution matches deployment

**Edge cases tested**
- [ ] Diverse inputs (long/short, edge of distribution)
- [ ] No NaN/Inf outputs
- [ ] Tested on the exact hardware/driver/runtime combination used in prod

**Rollback plan**
- [ ] Can revert to higher-precision model
- [ ] Monitoring catches accuracy/quality regressions
- [ ] A/B comparison against the unquantized baseline


## Skill Mastery Checklist

You have mastered quantization for inference when you can:

- [ ] Recognize when quantization is appropriate (CPU/edge/memory-constrained, LLM serving)
- [ ] Choose the correct type (dynamic vs static eager vs PT2E vs QAT) based on requirements
- [ ] Implement dynamic quantization in PyTorch using `torch.ao.quantization`
- [ ] Implement static quantization with proper calibration in eager mode and PT2E
- [ ] Pick the right LLM quantization technique from the Part 7 matrix for given hardware
- [ ] Read OCP MX / FP8 hardware support tables and match formats to GPU generation
- [ ] Validate accuracy trade-offs systematically (baseline vs quantized on the right eval)
- [ ] Benchmark on the actual target hardware
- [ ] Decide when NOT to quantize (already fast, accuracy-critical, no memory pressure)
- [ ] Migrate code from legacy `torch.quantization` to `torch.ao.quantization`
- [ ] Cross-reference `yzmir-llm-specialist/llm-inference-optimization.md` for LLM-specific generation-quality decisions

**Key insight:** Quantization is not magic — it's a systematic trade-off of precision for performance. The skill is matching the right approach to your specific model class, hardware, and accuracy budget.

---

Tooling and APIs current as of 2026-05; revisit quarterly.
