---
description: Systematic PyTorch profiling using 4-phase framework to identify bottlenecks
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[script_path] [--cpu|--gpu|--memory|--io]"
---

# PyTorch Profiling Command

You are profiling PyTorch code to identify performance bottlenecks. Follow the 4-phase framework.

## Core Principle

**Profile before optimizing. Measure, don't guess.** The bottleneck is rarely where you think it is.

## Phase 1: Establish Baseline

Before profiling, establish a reproducible baseline:

```python
import torch
import time

def benchmark_iteration(model, dataloader, device, num_warmup=3, num_iterations=10):
    """Establish reproducible baseline timing."""
    model.eval()

    # Warmup (critical for GPU timing)
    for i, batch in enumerate(dataloader):
        if i >= num_warmup:
            break
        with torch.no_grad():
            _ = model(batch.to(device))

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch.to(device))

        if device.type == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU

        times.append(time.perf_counter() - start)

    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
    }
```

**Critical GPU Timing Rule**: Always use `torch.cuda.synchronize()` or CUDA events for GPU timing. `time.time()` alone is meaningless for async GPU ops.

## Phase 2: Identify Bottleneck Type

Run the profiler to categorize the bottleneck:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

def profile_bottleneck_type(model, dataloader, device):
    """Identify whether bottleneck is CPU, GPU, memory, or I/O."""

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1)
    ) as prof:
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            with torch.no_grad():
                _ = model(batch.to(device))
            prof.step()

    # Print summary sorted by different metrics
    print("=== CPU Time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == 'cuda':
        print("\n=== CUDA Time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n=== Memory ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    return prof
```

### Bottleneck Classification

| Symptom | Likely Bottleneck | Next Step |
|---------|-------------------|-----------|
| High CPU time, low GPU time | CPU-bound | Check data loading, preprocessing |
| GPU utilization < 80% | Memory bandwidth or kernel launch | Check tensor sizes, batch size |
| Large gaps between GPU ops | CPU-GPU sync issues | Check `.item()`, `.numpy()` calls |
| High memory allocation time | Memory fragmentation | Check allocation patterns |

## Phase 3: Narrow to Component

Once you know the bottleneck type, drill into the specific component:

### For CPU-Bound (Data Loading)

```python
from torch.utils.data import DataLoader
import time

def profile_dataloader(dataloader, num_batches=20):
    """Profile data loading separately from model."""
    times = []

    start = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        times.append(time.perf_counter() - start)
        start = time.perf_counter()

    avg_time = sum(times) / len(times)
    print(f"Average batch load time: {avg_time*1000:.2f}ms")
    print(f"Recommended: num_workers={min(8, os.cpu_count())}, pin_memory=True")

    return times
```

### For GPU-Bound (Model Operations)

```python
def profile_by_layer(model, sample_input, device):
    """Profile each layer's contribution."""

    # PyTorch 2.9: Use torch.profiler with module-level granularity
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_modules=True  # Group by module
    ) as prof:
        with torch.no_grad():
            _ = model(sample_input.to(device))

    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=20
    ))
```

### For Memory-Bound

```python
def profile_memory_timeline(model, dataloader, device):
    """Track memory over time to find allocation patterns."""

    # PyTorch 2.0+ memory snapshot
    torch.cuda.memory._record_memory_history(max_entries=100000)

    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        with torch.no_grad():
            _ = model(batch.to(device))

    # Export for visualization
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    print("Snapshot saved to memory_snapshot.pickle")
    print("Visualize at: https://pytorch.org/memory_viz")
```

## Phase 4: Identify Specific Operations

Drill down to the exact operation causing issues:

```python
def profile_with_stack_trace(model, sample_input, device):
    """Get stack traces for expensive operations."""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            _ = model(sample_input.to(device))

    # Export Chrome trace for detailed timeline
    prof.export_chrome_trace("trace.json")
    print("Trace exported to trace.json")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")

    # Print operations with source locations
    for event in prof.key_averages():
        if event.cuda_time_total > 1000:  # > 1ms
            print(f"{event.key}: {event.cuda_time_total/1000:.2f}ms")
            if event.stack:
                print(f"  Stack: {event.stack[:3]}")
```

## PyTorch 2.9 Profiling Features

**Compile Profiling:**
```python
# Profile torch.compile overhead
model = torch.compile(model, mode="default")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # First run includes compilation
    _ = model(input)

print("First run (includes compile):")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

# Subsequent runs are faster
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    _ = model(input)

print("\nSubsequent runs:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
```

**TorchInductor Analysis:**
```python
# See what Inductor optimizes
import torch._dynamo
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.output_code = True

model = torch.compile(model)
_ = model(input)  # Check logs for generated Triton kernels
```

## Common Optimization Patterns

After identifying bottlenecks, apply these fixes:

| Bottleneck | Fix |
|------------|-----|
| Data loading | `num_workers=4+`, `pin_memory=True`, `prefetch_factor=2` |
| Small batch GPU underutil | Increase batch size or use `torch.compile` |
| CPU-GPU transfers | Batch transfers, avoid `.item()` in loops |
| Memory bandwidth | Use mixed precision (`torch.amp`) |
| Kernel launch overhead | `torch.compile(mode="reduce-overhead")` |

## Output Format

After profiling, provide:
1. **Baseline**: Initial timing (mean, std)
2. **Bottleneck Type**: CPU/GPU/Memory/I/O
3. **Specific Component**: Which layer/operation
4. **Recommended Fix**: With expected improvement
5. **Verification Command**: How to confirm fix worked
