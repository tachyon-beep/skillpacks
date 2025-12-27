---
description: Optimize ML model for production inference - quantization, pruning, batching, and hardware tuning
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[model_path]"
---

# Optimize Inference Command

Optimize an ML model for production inference through quantization, compression, and hardware tuning.

## Core Principle

**Optimization order: Hardware tuning → Batching → Quantization → Pruning. Start with cheapest wins, escalate to model changes only if needed.**

## Optimization Decision Tree

```
Is latency acceptable?
├─ YES → Done (don't over-optimize)
└─ NO → Profile to find bottleneck
    ├─ CPU-bound → Try GPU, or optimize CPU ops
    ├─ Memory-bound → Quantize, reduce batch
    └─ IO-bound → Optimize data loading

After hardware:
├─ Still slow? → Enable batching
└─ Still slow? → Quantize (INT8)
    └─ Still slow? → Prune or distill
```

## Optimization Techniques

### Level 1: Hardware Tuning (Free Performance)

```python
# Enable TensorFloat-32 on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN autotuning
torch.backends.cudnn.benchmark = True

# Use memory efficient attention (if applicable)
torch.nn.functional.scaled_dot_product_attention  # PyTorch 2.0+
```

**Expected gain: 10-30% latency reduction**

### Level 2: Batching

```python
# Dynamic batching for inference server
class DynamicBatcher:
    def __init__(self, model, max_batch=32, max_wait_ms=50):
        self.model = model
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = []

    async def predict(self, input_data):
        future = asyncio.Future()
        self.queue.append((input_data, future))

        if len(self.queue) >= self.max_batch:
            await self._process_batch()
        else:
            await asyncio.sleep(self.max_wait)
            if self.queue:
                await self._process_batch()

        return await future

    async def _process_batch(self):
        batch = self.queue[:self.max_batch]
        self.queue = self.queue[self.max_batch:]

        inputs = torch.stack([item[0] for item in batch])
        outputs = self.model(inputs)

        for i, (_, future) in enumerate(batch):
            future.set_result(outputs[i])
```

**Expected gain: 2-10x throughput improvement**

### Level 3: Quantization

```python
# Post-Training Quantization (PTQ) - fastest to implement
import torch.quantization

# Dynamic quantization (weights only)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Static quantization (weights + activations) - better speedup
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Run calibration data through model
for data in calibration_loader:
    model(data)
torch.quantization.convert(model, inplace=True)
```

**ONNX Runtime Quantization:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QInt8
)
```

**Expected gain: 2-4x speedup, 4x model size reduction**

### Level 4: Pruning

```python
# Structured pruning (remove entire channels/heads)
import torch.nn.utils.prune as prune

# Prune 30% of channels
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, 'weight', amount=0.3, n=2, dim=0)

# Make pruning permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')
```

**Expected gain: Variable, depends on sparsity tolerance**

### Level 5: Knowledge Distillation

```python
# Train smaller model to mimic larger model
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

**Expected gain: Can achieve 90%+ teacher accuracy with 10x smaller model**

## Benchmarking

```python
# Benchmark before and after optimization
import time
import torch

def benchmark_model(model, input_shape, num_iterations=100, warmup=10):
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_input)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            model(dummy_input)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations * 1000
    print(f"Average inference time: {avg_time:.2f} ms")
    return avg_time
```

## Output Format

```markdown
## Optimization Report: [Model Name]

### Baseline Performance

| Metric | Value |
|--------|-------|
| Inference time | [X ms] |
| Throughput | [X req/s] |
| Model size | [X MB] |
| Memory usage | [X MB] |

### Optimizations Applied

| Technique | Speedup | Accuracy Impact | Applied |
|-----------|---------|-----------------|---------|
| Hardware tuning | [X%] | None | Yes/No |
| Batching | [X%] | None | Yes/No |
| Quantization | [X%] | [-X%] | Yes/No |
| Pruning | [X%] | [-X%] | Yes/No |

### Final Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Inference time | [X ms] | [Y ms] | [Z%] |
| Throughput | [X req/s] | [Y req/s] | [Z%] |
| Model size | [X MB] | [Y MB] | [Z%] |
| Accuracy | [X%] | [Y%] | [-Z%] |

### Recommended Configuration

```python
[Optimized inference code]
```

### Trade-offs

- [Accuracy vs Speed decisions]
- [Hardware requirements]

### Next Steps

- [ ] Validate on production traffic
- [ ] A/B test optimized model
- [ ] Monitor for accuracy degradation
```

## Cross-Pack Discovery

```python
import glob

# For PyTorch optimization
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/plugin.json")
if pytorch_pack:
    print("For PyTorch profiling: use yzmir-pytorch-engineering")

# For neural architectures
arch_pack = glob.glob("plugins/yzmir-neural-architectures/plugin.json")
if arch_pack:
    print("For architecture optimization: use yzmir-neural-architectures")
```

## Load Detailed Guidance

For quantization:
```
Load skill: yzmir-ml-production:using-ml-production
Then read: quantization-for-inference.md
```

For compression:
```
Then read: model-compression-techniques.md
```
