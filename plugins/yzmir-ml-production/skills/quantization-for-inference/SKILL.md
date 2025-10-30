---
name: quantization-for-inference
description: Reduce model size and increase inference speed through post-training quantization (PTQ), quantization-aware training (QAT), and precision selection. Use when models are too slow or large for CPU/edge deployment.
---

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

## Core Principle

**Quantization trades precision for performance.**

Quantization converts high-precision numbers (FP32: 32 bits) to low-precision integers (INT8: 8 bits or INT4: 4 bits). This provides:
- **4-8× smaller model size** (fewer bits per parameter)
- **2-4× faster inference on CPU** (INT8 operations faster than FP32)
- **Small accuracy loss** (typically 0.5-1% for INT8)

**Formula:** Lower precision (FP32 → INT8 → INT4) = Smaller size + Faster inference + More accuracy loss

The skill is choosing the **right precision for your accuracy tolerance**.

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
- ✅ 4× size reduction (weights are 75% of model size)
- ✅ 1.2-1.5× CPU speedup (modest, because activations still FP32)
- ✅ Minimal accuracy loss (~0.2-0.5%)
- ✅ No calibration data needed

**Limitations:**
- ⚠️ Limited CPU speedup (activations still FP32)
- ⚠️ Not optimal for edge devices needing maximum performance

**PyTorch implementation:**

```python
import torch
import torch.quantization

# WHY: Dynamic quantization is simplest - just one function call
# No calibration data needed because activations stay FP32
model = torch.load('model.pth')
model.eval()  # WHY: Must be in eval mode (no batchnorm updates)

# WHY: Specify which layers to quantize (Linear, LSTM, etc.)
# These layers benefit most from quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},  # WHY: Quantize Linear layers only
    dtype=torch.qint8  # WHY: INT8 is standard precision
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized_dynamic.pth')

# Verify size reduction
original_size = os.path.getsize('model.pth') / (1024 ** 2)  # MB
quantized_size = os.path.getsize('model_quantized_dynamic.pth') / (1024 ** 2)
print(f"Original: {original_size:.1f}MB → Quantized: {quantized_size:.1f}MB")
print(f"Size reduction: {original_size / quantized_size:.1f}×")
```

**Example use case:** BERT classification model where primary goal is reducing size from 440MB to 110MB for easier deployment.

---

### Type 2: Static Quantization (Post-Training Quantization)

**What it does:** Quantizes both weights and activations to INT8.

**When to use:**
- Need maximum CPU speedup (2-4×)
- Deploying to CPU servers or edge devices
- Can afford calibration step (5-10 minutes)
- Primary goal is inference speed

**Benefits:**
- ✅ 4× size reduction (same as dynamic)
- ✅ 2-4× CPU speedup (both weights and activations INT8)
- ✅ No retraining required (post-training)
- ✅ Acceptable accuracy loss (~0.5-1%)

**Requirements:**
- ⚠️ Needs calibration data (100-1000 samples from validation set)
- ⚠️ Slightly more complex setup than dynamic

**PyTorch implementation:**

```python
import torch
import torch.quantization

def calibrate_model(model, calibration_loader):
    """
    Calibrate model by running representative data through it.

    WHY: Static quantization needs to know activation ranges.
    Calibration finds min/max values for each activation layer.

    Args:
        model: Model in eval mode with quantization stubs
        calibration_loader: DataLoader with 100-1000 samples
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
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

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
torch.quantization.convert(model, inplace=True)

# Save quantized model
torch.save(model.state_dict(), 'model_quantized_static.pth')

# Benchmark speed improvement
import time

def benchmark(model, data, num_iterations=100):
    """WHY: Warm up model first, then measure average latency."""
    model.eval()
    # Warm up (first few iterations slower)
    for _ in range(10):
        model(data)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(data)
    end = time.time()
    return (end - start) / num_iterations * 1000  # ms per inference

test_data = torch.randn(1, 3, 224, 224)  # Example input

baseline_latency = benchmark(original_model, test_data)
quantized_latency = benchmark(model, test_data)

print(f"Baseline: {baseline_latency:.2f}ms")
print(f"Quantized: {quantized_latency:.2f}ms")
print(f"Speedup: {baseline_latency / quantized_latency:.2f}×")
```

**Example use case:** ResNet50 image classifier for CPU inference - need <5ms latency, achieve 4ms with static quantization (vs 15ms baseline).

---

### Type 3: Quantization-Aware Training (QAT)

**What it does:** Simulates quantization during training to minimize accuracy loss.

**When to use:**
- Static quantization accuracy loss too large (>2%)
- Need best possible accuracy with INT8
- Can afford retraining (hours to days)
- Critical production system with strict accuracy requirements

**Benefits:**
- ✅ Best accuracy (~0.1-0.3% loss vs 0.5-1% for static)
- ✅ 4× size reduction (same as dynamic/static)
- ✅ 2-4× CPU speedup (same as static)

**Limitations:**
- ⚠️ Requires retraining (most expensive option)
- ⚠️ Takes hours to days depending on model size
- ⚠️ More complex implementation

**PyTorch implementation:**

```python
import torch
import torch.quantization

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
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Step 2: Train with quantization-aware training
# WHY: Model learns to compensate for quantization errors
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # WHY: Low LR for fine-tuning
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5  # WHY: Usually 5-10 epochs sufficient for QAT fine-tuning
for epoch in range(num_epochs):
    model = train_one_epoch_qat(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} complete")

# Step 3: Convert to quantized model
model.eval()
torch.quantization.convert(model, inplace=True)

# Save QAT quantized model
torch.save(model.state_dict(), 'model_quantized_qat.pth')
```

**Example use case:** Medical imaging model where accuracy is critical - static quantization gives 2% accuracy loss, QAT reduces to 0.3%.

---

## Part 2: Quantization Type Decision Matrix

| Type | Complexity | Calibration | Retraining | Size Reduction | CPU Speedup | Accuracy Loss |
|------|-----------|-------------|------------|----------------|-------------|---------------|
| **Dynamic** | Low | No | No | 4× | 1.2-1.5× | ~0.2-0.5% |
| **Static** | Medium | Yes | No | 4× | 2-4× | ~0.5-1% |
| **QAT** | High | Yes | Yes | 4× | 2-4× | ~0.1-0.3% |

**Decision flow:**
1. Start with **dynamic quantization**: Simplest, verify quantization helps
2. Upgrade to **static quantization**: If need more speedup, can afford calibration
3. Use **QAT**: Only if accuracy loss from static too large (rare)

**Why this order?** Incremental cost. Dynamic is free (5 minutes), static is cheap (15 minutes), QAT is expensive (hours/days). Don't pay for QAT unless you need it.

---

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
- ✅ **Use validation set samples** (matches training distribution)
- ❌ Don't use random images from internet (different distribution)
- ❌ Don't use single image repeated (insufficient coverage)
- ❌ Don't use training set that doesn't match deployment (distribution shift)

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

    Args:
        val_dataset: Full validation dataset
        num_samples: Number of calibration samples (default 1000)

    Returns:
        Calibration dataset subset
    """
    # WHY: Random selection ensures diversity
    # Stratified sampling can help ensure class coverage
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    calibration_dataset = torch.utils.data.Subset(val_dataset, indices)

    return calibration_dataset

# Example: Select 1000 random samples from validation set
calibration_dataset = select_calibration_data(val_dataset, num_samples=1000)
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=False  # WHY: Order doesn't matter for calibration
)
```

### Common Calibration Pitfalls

**Pitfall 1: Using wrong data distribution**
- ❌ "Random images from internet" for ImageNet-trained model
- ✅ Use ImageNet validation set samples

**Pitfall 2: Too few samples**
- ❌ 10 samples (insufficient coverage of activation ranges)
- ✅ 100-1000 samples (good coverage)

**Pitfall 3: Using training data that doesn't match deployment**
- ❌ Calibrate on sunny outdoor images, deploy on indoor images
- ✅ Calibrate on data matching deployment distribution

**Pitfall 4: Skipping calibration validation**
- ❌ Calibrate once, assume it works
- ✅ Validate accuracy after calibration to verify ranges are good

---

## Part 4: Precision Selection (INT8 vs INT4 vs FP16)

### Precision Spectrum

| Precision | Bits | Size vs FP32 | Speedup (CPU) | Typical Accuracy Loss |
|-----------|------|--------------|---------------|----------------------|
| **FP32** | 32 | 1× | 1× | 0% (baseline) |
| **FP16** | 16 | 2× | 1.5× | <0.1% |
| **INT8** | 8 | 4× | 2-4× | 0.5-1% |
| **INT4** | 4 | 8× | 4-8× | 1-3% |

**Trade-off:** Lower precision = Smaller size + Faster inference + More accuracy loss

### When to Use Each Precision

**FP16 (Half Precision):**
- GPU inference (Tensor Cores optimized for FP16)
- Need minimal accuracy loss (<0.1%)
- Size reduction secondary concern
- **Example:** Large language models on GPU

**INT8 (Standard Quantization):**
- CPU inference (INT8 operations fast on CPU)
- Edge device deployment
- Good balance of size/speed/accuracy
- **Most common choice** for production deployment
- **Example:** Image classification on mobile devices

**INT4 (Aggressive Quantization):**
- Extremely memory-constrained (e.g., 1GB mobile devices)
- Can tolerate larger accuracy loss (1-3%)
- Need maximum size reduction (8×)
- **Use sparingly** - accuracy risk high
- **Example:** Large language models (LLaMA-7B: 13GB → 3.5GB)

### Decision Flow

```python
def choose_precision(accuracy_tolerance, deployment_target):
    """
    Choose quantization precision based on requirements.

    WHY: Different precisions for different constraints.
    INT8 is default, FP16 for GPU, INT4 for extreme memory constraints.
    """
    if accuracy_tolerance < 0.1:
        return "FP16"  # Minimal accuracy loss required
    elif deployment_target == "GPU":
        return "FP16"  # GPU optimized for FP16
    elif deployment_target in ["CPU", "edge"]:
        return "INT8"  # CPU optimized for INT8
    elif deployment_target == "extreme_edge" and accuracy_tolerance > 1:
        return "INT4"  # Only if can tolerate 1-3% loss
    else:
        return "INT8"  # Default safe choice
```

---

## Part 5: ONNX Quantization (Cross-Framework)

**When to use:** Deploying to ONNX Runtime (CPU/edge devices) or need cross-framework compatibility.

### ONNX Static Quantization

```python
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader
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
        """WHY: Called by ONNX to get next calibration batch."""
        try:
            data, _ = next(self.iterator)
            return {"input": data.numpy()}  # WHY: Return dict of input name → data
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
    opset_version=13  # WHY: ONNX opset 13+ supports quantization ops
)

# Step 2: Prepare calibration data
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=1,  # WHY: ONNX calibration uses batch size 1
    shuffle=False
)
calibration_reader = CalibrationDataReaderWrapper(calibration_loader)

# Step 3: Quantize ONNX model
quantize_static(
    'model.onnx',
    'model_quantized.onnx',
    calibration_data_reader=calibration_reader,
    quant_format='QDQ'  # WHY: QDQ format compatible with most backends
)

# Step 4: Benchmark ONNX quantized model
import time

session = onnxruntime.InferenceSession('model_quantized.onnx')
input_name = session.get_inputs()[0].name

test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warm up
for _ in range(10):
    session.run(None, {input_name: test_data})

# Benchmark
start = time.time()
for _ in range(100):
    session.run(None, {input_name: test_data})
end = time.time()

latency = (end - start) / 100 * 1000  # ms per inference
print(f"ONNX Quantized latency: {latency:.2f}ms")
```

**ONNX advantages:**
- Cross-framework (works with PyTorch, TensorFlow, etc.)
- Optimized ONNX Runtime for CPU inference
- Good hardware backend support (x86, ARM)

---

## Part 6: Accuracy Validation (Critical Step)

### Why Accuracy Validation Matters

Quantization is **lossy compression**. Must measure accuracy impact:
- Some models tolerate quantization well (<0.5% loss)
- Some models sensitive to quantization (>2% loss)
- Some layers more sensitive than others
- **Can't assume quantization is safe without measuring**

### Validation Methodology

```python
def validate_quantization(original_model, quantized_model, val_loader):
    """
    Validate quantization by comparing accuracy.

    WHY: Quantization is lossy - must measure impact.
    Compare baseline vs quantized on same validation set.

    Returns:
        dict with baseline_acc, quantized_acc, accuracy_loss
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
        'acceptable': accuracy_loss < 2.0  # WHY: <2% loss usually acceptable
    }

# Example validation
results = validate_quantization(original_model, quantized_model, val_loader)
print(f"Baseline accuracy: {results['baseline_acc']:.2f}%")
print(f"Quantized accuracy: {results['quantized_acc']:.2f}%")
print(f"Accuracy loss: {results['accuracy_loss']:.2f}%")
print(f"Acceptable: {results['acceptable']}")

# Decision logic
if results['acceptable']:
    print("✅ Quantization acceptable - deploy quantized model")
else:
    print("❌ Accuracy loss too large - try QAT or reconsider quantization")
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
- Medical/safety-critical: <0.5% loss required (use QAT)

---

## Part 7: LLM Quantization (GPTQ, AWQ)

**Note:** This skill covers general quantization. For LLM-specific optimization (GPTQ, AWQ, KV cache, etc.), see the `llm-inference-optimization` skill in the llm-specialist pack.

### LLM Quantization Overview

**Why LLMs need quantization:**
- Very large (7B parameters = 13GB in FP16)
- Memory-bound inference (limited by VRAM)
- INT4 quantization: 13GB → 3.5GB (fits in consumer GPUs)

**LLM-specific quantization methods:**
- **GPTQ:** Post-training quantization optimized for LLMs
- **AWQ:** Activation-aware weight quantization (better quality than GPTQ)
- **Both:** Achieve INT4 with <0.5 perplexity increase

### When to Use LLM Quantization

✅ **Use when:**
- Deploying LLMs locally (consumer GPUs)
- Memory-constrained (need to fit in 12GB/24GB VRAM)
- Cost-sensitive (smaller models cheaper to host)
- Latency-sensitive (smaller models faster to load)

❌ **Don't use when:**
- Have sufficient GPU memory for FP16
- Accuracy critical (medical, legal applications)
- Already using API (OpenAI, Anthropic) - they handle optimization

### LLM Quantization References

For detailed LLM quantization:
- **See skill:** `llm-inference-optimization` (llm-specialist pack)
- **Covers:** GPTQ, AWQ, KV cache optimization, token streaming
- **Tools:** llama.cpp, vLLM, text-generation-inference

**Quick reference (defer to llm-specialist for details):**

```python
# GPTQ quantization (example - see llm-specialist for full details)
from transformers import AutoModelForCausalLM, GPTQConfig

# WHY: GPTQ optimizes layer-wise for minimal perplexity increase
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Result: 13GB → 3.5GB, <0.5 perplexity increase
```

---

## Part 8: When NOT to Quantize

### Scenario 1: Already Fast Enough

**Example:** MobileNetV2 (14MB, 3ms CPU latency)
- Quantization: 14MB → 4MB, 3ms → 2ms
- **Benefit:** 10MB saved, 1ms faster
- **Cost:** Calibration, validation, testing, debugging
- **Decision:** Not worth effort unless specific requirement

**Rule:** If current performance meets requirements, don't optimize.

### Scenario 2: GPU-Only Deployment with No Memory Constraints

**Example:** ResNet50 on Tesla V100 with 32GB VRAM
- Quantization: 1.5-2× GPU speedup (modest)
- FP32 already fast on GPU (Tensor Cores optimized)
- No memory pressure (plenty of VRAM)
- **Decision:** Focus on other bottlenecks (data loading, I/O)

**Rule:** Quantization is most beneficial for CPU inference and memory-constrained GPU.

### Scenario 3: Accuracy-Critical Applications

**Example:** Medical diagnosis model where misdiagnosis has severe consequences
- Quantization introduces accuracy loss (even if small)
- Risk not worth benefit
- **Decision:** Keep FP32, optimize other parts (batching, caching)

**Rule:** Safety-critical systems should avoid lossy compression unless thoroughly validated.

### Scenario 4: Prototyping Phase

**Example:** Early development, trying different architectures
- Quantization is optimization - premature at prototype stage
- Focus on getting model working first
- **Decision:** Defer quantization until production deployment

**Rule:** Don't optimize until you need to (Knuth: "Premature optimization is root of all evil").

---

## Part 9: Quantization Benchmarks (Expected Results)

### Image Classification (ResNet50, ImageNet)

| Metric | FP32 Baseline | Dynamic INT8 | Static INT8 | QAT INT8 |
|--------|---------------|--------------|-------------|----------|
| Size | 98MB | 25MB (4×) | 25MB (4×) | 25MB (4×) |
| CPU Latency | 15ms | 12ms (1.25×) | 4ms (3.75×) | 4ms (3.75×) |
| Top-1 Accuracy | 76.1% | 75.9% (0.2% loss) | 75.3% (0.8% loss) | 75.9% (0.2% loss) |

**Insight:** Static quantization gives 3.75× speedup with acceptable 0.8% accuracy loss.

### Object Detection (YOLOv5s, COCO)

| Metric | FP32 Baseline | Static INT8 | QAT INT8 |
|--------|---------------|-------------|----------|
| Size | 14MB | 4MB (3.5×) | 4MB (3.5×) |
| CPU Latency | 45ms | 15ms (3×) | 15ms (3×) |
| mAP@0.5 | 37.4% | 36.8% (0.6% loss) | 37.2% (0.2% loss) |

**Insight:** QAT gives better accuracy (0.2% vs 0.6% loss) with same speedup.

### NLP Classification (BERT-base, GLUE)

| Metric | FP32 Baseline | Dynamic INT8 | Static INT8 |
|--------|---------------|--------------|-------------|
| Size | 440MB | 110MB (4×) | 110MB (4×) |
| CPU Latency | 35ms | 28ms (1.25×) | 12ms (2.9×) |
| Accuracy | 93.5% | 93.2% (0.3% loss) | 92.8% (0.7% loss) |

**Insight:** Static quantization gives 2.9× speedup but dynamic sufficient if speedup not critical.

### LLM Inference (LLaMA-7B)

| Metric | FP16 Baseline | GPTQ INT4 | AWQ INT4 |
|--------|---------------|-----------|----------|
| Size | 13GB | 3.5GB (3.7×) | 3.5GB (3.7×) |
| First Token Latency | 800ms | 250ms (3.2×) | 230ms (3.5×) |
| Perplexity | 5.68 | 5.82 (0.14 increase) | 5.77 (0.09 increase) |

**Insight:** AWQ gives better quality than GPTQ with similar speedup.

---

## Part 10: Common Pitfalls and Solutions

### Pitfall 1: Skipping Accuracy Validation

**Issue:** Deploy quantized model without measuring accuracy impact.
**Risk:** Discover accuracy degradation in production (too late).
**Solution:** Always validate accuracy on representative data before deployment.

```python
# ❌ WRONG: Deploy without validation
quantized_model = quantize(model)
deploy(quantized_model)  # Hope it works!

# ✅ RIGHT: Validate before deployment
quantized_model = quantize(model)
results = validate_accuracy(original_model, quantized_model, val_loader)
if results['acceptable']:
    deploy(quantized_model)
else:
    print("Accuracy loss too large - try QAT")
```

### Pitfall 2: Using Wrong Calibration Data

**Issue:** Calibrate with random/unrepresentative data.
**Risk:** Activation ranges wrong → accuracy collapses.
**Solution:** Use 100-1000 samples from validation set matching deployment distribution.

```python
# ❌ WRONG: Random images from internet
calibration_data = download_random_images()

# ✅ RIGHT: Samples from validation set
calibration_data = torch.utils.data.Subset(val_dataset, range(1000))
```

### Pitfall 3: Choosing Wrong Quantization Type

**Issue:** Use dynamic quantization when need static speedup.
**Risk:** Get 1.2× speedup instead of 3× speedup.
**Solution:** Match quantization type to requirements (dynamic for size, static for speed).

```python
# ❌ WRONG: Use dynamic when need speed
if need_fast_cpu_inference:
    quantized_model = torch.quantization.quantize_dynamic(model)  # Only 1.2× speedup

# ✅ RIGHT: Use static for speed
if need_fast_cpu_inference:
    model = prepare_and_calibrate(model, calibration_data)
    quantized_model = torch.quantization.convert(model)  # 2-4× speedup
```

### Pitfall 4: Quantizing GPU-Only Deployments

**Issue:** Quantize model for GPU inference without memory pressure.
**Risk:** Effort not worth modest 1.5-2× GPU speedup.
**Solution:** Only quantize GPU if memory-constrained (multiple models in VRAM).

```python
# ❌ WRONG: Quantize for GPU with no memory issue
if deployment_target == "GPU" and have_plenty_of_memory:
    quantized_model = quantize(model)  # Wasted effort

# ✅ RIGHT: Skip quantization if not needed
if deployment_target == "GPU" and have_plenty_of_memory:
    deploy(model)  # Keep FP32, focus on other optimizations
```

### Pitfall 5: Over-Quantizing (INT4 When INT8 Sufficient)

**Issue:** Use aggressive INT4 quantization when INT8 would suffice.
**Risk:** Larger accuracy loss than necessary.
**Solution:** Start with INT8 (standard), only use INT4 if extreme memory constraints.

```python
# ❌ WRONG: Jump to INT4 without trying INT8
quantized_model = quantize(model, precision="INT4")  # 2-3% accuracy loss

# ✅ RIGHT: Start with INT8, only use INT4 if needed
quantized_model_int8 = quantize(model, precision="INT8")  # 0.5-1% accuracy loss
if model_still_too_large:
    quantized_model_int4 = quantize(model, precision="INT4")
```

### Pitfall 6: Assuming All Layers Quantize Equally

**Issue:** Quantize all layers uniformly, but some layers more sensitive.
**Risk:** Accuracy loss dominated by few sensitive layers.
**Solution:** Use mixed precision - keep sensitive layers in FP32/INT8, quantize others to INT4.

```python
# ✅ ADVANCED: Mixed precision quantization
# Keep first/last layers in higher precision, quantize middle layers aggressively
from torch.quantization import QConfigMapping

qconfig_mapping = QConfigMapping()
qconfig_mapping.set_global(get_default_qconfig('fbgemm'))  # INT8 default
qconfig_mapping.set_module_name('model.layer1', None)  # Keep first layer FP32
qconfig_mapping.set_module_name('model.layer10', None)  # Keep last layer FP32

model = quantize_with_qconfig(model, qconfig_mapping)
```

---

## Part 11: Decision Framework Summary

### Step 1: Recognize Quantization Need

**Symptoms:**
- Model too slow on CPU (>10ms when need <5ms)
- Model too large for edge devices (>50MB)
- Deploying to CPU/edge (not GPU)
- Need to reduce hosting costs

**If YES → Proceed to Step 2**
**If NO → Don't quantize, focus on other optimizations**

### Step 2: Choose Quantization Type

```
Start with Dynamic:
├─ Sufficient? (meets latency/size requirements)
│  ├─ YES → Deploy dynamic quantized model
│  └─ NO → Proceed to Static
│
Static Quantization:
├─ Sufficient? (meets latency/size + accuracy acceptable)
│  ├─ YES → Deploy static quantized model
│  └─ NO → Accuracy loss >2%
│     │
│     └─ Proceed to QAT
│
QAT:
├─ Train with quantization awareness
└─ Achieves <1% accuracy loss → Deploy
```

### Step 3: Calibrate (if Static/QAT)

**Calibration data:**
- Source: Validation set (representative samples)
- Size: 100-1000 samples
- Characteristics: Match deployment distribution

**Calibration process:**
1. Select samples from validation set
2. Run through model to collect activation ranges
3. Validate accuracy after calibration
4. If accuracy loss >2%, try different calibration data or QAT

### Step 4: Validate Accuracy

**Required measurements:**
- Baseline accuracy (FP32)
- Quantized accuracy (INT8/INT4)
- Accuracy loss (baseline - quantized)
- Acceptable threshold (typically <2%)

**Decision:**
- If accuracy loss <2% → Deploy
- If accuracy loss >2% → Try QAT or reconsider quantization

### Step 5: Benchmark Performance

**Required measurements:**
- Model size (MB): baseline vs quantized
- Inference latency (ms): baseline vs quantized
- Throughput (requests/sec): baseline vs quantized

**Verify expected results:**
- Size: 4× reduction (FP32 → INT8)
- CPU speedup: 2-4× (static quantization)
- GPU speedup: 1.5-2× (if applicable)

---

## Part 12: Production Deployment Checklist

Before deploying quantized model to production:

**✅ Accuracy Validated**
- [ ] Baseline accuracy measured on validation set
- [ ] Quantized accuracy measured on same validation set
- [ ] Accuracy loss within acceptable threshold (<2%)
- [ ] Validated on representative production data

**✅ Performance Benchmarked**
- [ ] Size reduction measured (expect 4× for INT8)
- [ ] Latency improvement measured (expect 2-4× CPU)
- [ ] Throughput improvement measured
- [ ] Performance meets requirements

**✅ Calibration Verified** (if static/QAT)
- [ ] Used representative samples from validation set (not random data)
- [ ] Used sufficient calibration data (100-1000 samples)
- [ ] Calibration data matches deployment distribution

**✅ Edge Cases Tested**
- [ ] Tested on diverse inputs (bright/dark images, long/short text)
- [ ] Validated numerical stability (no NaN/Inf outputs)
- [ ] Tested inference on target hardware (CPU/GPU/edge device)

**✅ Rollback Plan**
- [ ] Can easily revert to FP32 model if issues found
- [ ] Monitoring in place to detect accuracy degradation
- [ ] A/B testing plan to compare FP32 vs quantized

---

## Skill Mastery Checklist

You have mastered quantization for inference when you can:

- [ ] Recognize when quantization is appropriate (CPU/edge deployment, size/speed issues)
- [ ] Choose correct quantization type (dynamic vs static vs QAT) based on requirements
- [ ] Implement dynamic quantization in PyTorch (5 lines of code)
- [ ] Implement static quantization with proper calibration (20 lines of code)
- [ ] Select appropriate calibration data (validation set, 100-1000 samples)
- [ ] Validate accuracy trade-offs systematically (baseline vs quantized)
- [ ] Benchmark performance improvements (size, latency, throughput)
- [ ] Decide when NOT to quantize (GPU-only, already fast, accuracy-critical)
- [ ] Debug quantization issues (accuracy collapse, wrong speedup, numerical instability)
- [ ] Deploy quantized models to production with confidence

**Key insight:** Quantization is not magic - it's a systematic trade-off of precision for performance. The skill is matching the right quantization approach to your specific requirements.
