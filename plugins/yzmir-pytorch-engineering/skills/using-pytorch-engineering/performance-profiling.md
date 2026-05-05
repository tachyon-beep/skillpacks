
# Performance Profiling and Bottleneck Analysis

## Overview

**Core Principle:** Optimization without measurement is guessing. Profile systematically (whole → component → operation) using the right tools to identify actual bottlenecks before attempting fixes. 90% of runtime is usually in 10% of code - find that 10% with profiling, not intuition.

Performance issues stem from: data loading bottlenecks (CPU-bound), inefficient operations (GPU-bound), memory bandwidth limits (memory-bound), or I/O bottlenecks. Profiling reveals which category applies. Guessing leads to optimizing the wrong thing, wasting hours on marginal improvements while real bottleneck remains.

## When to Use

**Use this skill when:**
- Training or inference slower than expected
- Need to identify performance bottleneck in PyTorch code
- High GPU memory usage, need to understand what's using memory
- Evaluating whether optimization actually improved performance
- Debugging low GPU utilization issues
- Comparing performance of different implementations
- Need to profile specific operations or model components

**Don't use when:**
- Performance is already acceptable (no problem to solve)
- Architecture design questions (use module-design-patterns)
- Debugging correctness issues (use debugging-techniques)
- Memory leaks (use tensor-operations-and-memory)

**Symptoms triggering this skill:**
- "Training is slower than expected"
- "Low GPU utilization but training still slow"
- "Which part of my model is the bottleneck?"
- "Does this optimization actually help?"
- "Memory usage is high, what's using it?"
- "First iteration much slower than subsequent ones"


## Systematic Profiling Methodology

### The Four-Phase Framework

**Phase 1: Establish Baseline**
- Define metric (throughput, latency, memory)
- Measure end-to-end performance
- Set improvement target
- Document measurement conditions

**Phase 2: Identify Bottleneck Type**
- CPU-bound vs GPU-bound vs I/O-bound vs memory-bound
- Check GPU utilization (nvidia-smi)
- Profile data loading separately from computation
- Determine which component to investigate

**Phase 3: Narrow to Component**
- Profile at coarse granularity
- Identify which phase is slow (forward/backward/optimizer/data loading)
- Focus profiling on bottleneck component
- Use iterative narrowing

**Phase 4: Identify Operation**
- Profile bottleneck component in detail
- Examine both table view and trace view
- Find specific operation or pattern
- Measure improvement after fix

**Critical Rule:** ALWAYS work through phases in order. Don't jump to Phase 4 without Phases 1-3.


### Phase 1: Establish Baseline

**Step 1: Define Performance Metric**

```python
# Choose the right metric for your use case:

# Throughput (samples/second) - for training
def measure_throughput(model, dataloader, num_batches=100):
    model.train()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_samples = 0
    start.record()

    for i, (data, target) in enumerate(dataloader):
        if i >= num_batches:
            break
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_samples += data.size(0)

    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    throughput = total_samples / (elapsed_ms / 1000.0)
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Time per batch: {elapsed_ms / num_batches:.2f} ms")
    return throughput

# Latency (time per sample) - for inference
def measure_latency(model, sample_input, num_iterations=100, warmup=10):
    model.eval()
    sample_input = sample_input.cuda()

    # Warmup (CRITICAL - don't skip!)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = model(sample_input)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    # Report statistics (not just average!)
    import numpy as np
    latencies = np.array(latencies)
    print(f"Latency - Mean: {latencies.mean():.2f} ms, "
          f"Std: {latencies.std():.2f} ms, "
          f"Median: {np.median(latencies):.2f} ms, "
          f"P95: {np.percentile(latencies, 95):.2f} ms, "
          f"P99: {np.percentile(latencies, 99):.2f} ms")
    return latencies

# Memory usage (peak GB)
def measure_memory(model, sample_batch):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run one iteration
    output = model(sample_batch)
    loss = criterion(output, target)
    loss.backward()

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory: {peak_memory:.2f} GB")
    return peak_memory
```

**Why this matters:**
- Without baseline, can't measure improvement
- Need statistics (mean, std, percentiles), not just average
- Must use CUDA Events for GPU timing (not `time.time()`)
- Warmup critical to exclude JIT compilation overhead


**Step 2: Document Measurement Conditions**

```python
# Record all relevant configuration
profiling_config = {
    'model': model.__class__.__name__,
    'batch_size': 32,
    'input_shape': (3, 224, 224),
    'device': 'cuda:0',
    'dtype': 'float16' if using_amp else 'float32',
    'mode': 'train' or 'eval',
    'num_workers': dataloader.num_workers,
    'cudnn_benchmark': torch.backends.cudnn.benchmark,
    'gpu': torch.cuda.get_device_name(0),
}

print(json.dumps(profiling_config, indent=2))
```

**Why this matters:**
- Performance changes with configuration
- Need to reproduce results
- Comparing different runs requires same conditions
- Document before optimizing, re-measure after


### Phase 2: Identify Bottleneck Type

**Step 1: Check GPU Utilization**

```bash
# In terminal, monitor GPU utilization in real-time
nvidia-smi dmon -s u -i 0 -d 1

# Or within Python
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
gpu_util, mem_used = result.stdout.strip().split(',')
print(f"GPU Utilization: {gpu_util}%, Memory: {mem_used} MB")
```

**Interpretation:**

| GPU Utilization | Likely Bottleneck | Next Step |
|----------------|-------------------|-----------|
| < 70% | CPU-bound (data loading, preprocessing) | Profile data loading |
| > 90% | GPU-bound (computation) | Profile model operations |
| 50-80% | Mixed or memory-bound | Check memory bandwidth |

**Why this matters:** GPU utilization tells you WHERE to look. If GPU isn't saturated, optimizing GPU operations won't help.


**Step 2: Profile Data Loading vs Computation**

```python
import time

def profile_dataloader_vs_model(model, dataloader, num_batches=50):
    """Separate data loading time from model computation time"""
    model.train()

    data_times = []
    compute_times = []

    batch_iterator = iter(dataloader)

    for i in range(num_batches):
        # Time data loading
        data_start = time.time()
        data, target = next(batch_iterator)
        data_end = time.time()
        data_times.append(data_end - data_start)

        # Time computation
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        torch.cuda.synchronize()

        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)

        compute_start.record()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        compute_end.record()

        torch.cuda.synchronize()
        compute_times.append(compute_start.elapsed_time(compute_end))

    import numpy as np
    avg_data_time = np.mean(data_times) * 1000  # to ms
    avg_compute_time = np.mean(compute_times)

    print(f"Avg data loading time: {avg_data_time:.2f} ms")
    print(f"Avg computation time: {avg_compute_time:.2f} ms")
    print(f"Data loading is {avg_data_time/avg_compute_time:.1f}x "
          f"{'slower' if avg_data_time > avg_compute_time else 'faster'} than compute")

    if avg_data_time > avg_compute_time:
        print("⚠️ BOTTLENECK: Data loading (CPU-bound)")
        print("   Solutions: Increase num_workers, use pin_memory=True, "
              "move preprocessing to GPU")
    else:
        print("✅ Data loading is fast enough. Bottleneck is in model computation.")

    return avg_data_time, avg_compute_time
```

**Why this matters:**
- If data loading > computation time, GPU is starved (increase workers)
- If computation > data loading, GPU is bottleneck (optimize model)
- Common mistake: Optimizing model when data loading is the bottleneck


**Step 3: Determine Bottleneck Category**

```python
def diagnose_bottleneck_type(model, dataloader):
    """Systematic bottleneck categorization"""

    # 1. Check GPU utilization
    print("=== GPU Utilization Check ===")
    # Run training for a bit while monitoring GPU
    # If GPU util < 70% → CPU-bound
    # If GPU util > 90% → GPU-bound

    # 2. Check memory bandwidth
    print("\n=== Memory Bandwidth Check ===")
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

    # Look for high memory-bound ops
    events = prof.key_averages()
    for evt in events:
        if evt.cuda_time_total > 0:
            # If many large tensor ops with low FLOPS → memory-bound
            pass

    # 3. Profile phases
    print("\n=== Phase Profiling ===")
    times = profile_training_phases(model, next(iter(dataloader)))

    # Interpret results
    print("\n=== Diagnosis ===")
    if times['data_loading'] > times['forward'] + times['backward']:
        print("BOTTLENECK: CPU-bound (data loading)")
        print("Action: Increase num_workers, enable pin_memory, cache data")
    elif times['forward'] > times['backward'] * 2:
        print("BOTTLENECK: GPU-bound (forward pass)")
        print("Action: Profile forward pass operations")
    elif times['backward'] > times['forward'] * 2:
        print("BOTTLENECK: GPU-bound (backward pass)")
        print("Action: Profile backward pass, check gradient checkpointing")
    else:
        print("BOTTLENECK: Mixed or memory-bound")
        print("Action: Deep profiling needed")
```


### Phase 3: Narrow to Component

**Step 1: Coarse-Grained Profiling**

```python
from torch.profiler import profile, ProfilerActivity, schedule

def profile_training_step(model, dataloader, num_steps=10):
    """Profile one training step to identify bottleneck phase"""

    # Use schedule to reduce profiling overhead
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        for step, (data, target) in enumerate(dataloader):
            if step >= num_steps:
                break

            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            # Forward
            output = model(data)
            loss = criterion(output, target)

            # Backward
            loss.backward()

            # Optimizer
            optimizer.step()

            prof.step()  # Notify profiler of step boundary

    # Print summary
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_src_column_width=80
    ))

    # Export trace for visualization
    print("\n✅ Trace exported to ./profiler_logs")
    print("   View in Chrome: chrome://tracing (load trace.json)")
    print("   Or TensorBoard: tensorboard --logdir=./profiler_logs")

    return prof
```

**Understanding the schedule:**
- `wait=1`: Skip first iteration (cold start)
- `warmup=2`: Next 2 iterations for warmup (no profiling overhead)
- `active=5`: Profile these 5 iterations
- `repeat=1`: Do this cycle once

**Why this matters:**
- Profiling has overhead - don't profile every iteration
- Schedule controls when profiling is active
- Warmup prevents including JIT compilation in measurements


**Step 2: Phase-Level Timing**

```python
from torch.profiler import record_function

def profile_training_phases(model, batch, target):
    """Time each phase of training separately"""

    data, target = batch.cuda(), target.cuda()
    optimizer.zero_grad()

    torch.cuda.synchronize()

    # Profile each phase
    phases = {}

    # Forward pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with record_function("forward_pass"):
        start.record()
        output = model(data)
        loss = criterion(output, target)
        end.record()
        torch.cuda.synchronize()
        phases['forward'] = start.elapsed_time(end)

    # Backward pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with record_function("backward_pass"):
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        phases['backward'] = start.elapsed_time(end)

    # Optimizer step
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with record_function("optimizer_step"):
        start.record()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        phases['optimizer'] = start.elapsed_time(end)

    # Print breakdown
    total = sum(phases.values())
    print("Phase Breakdown:")
    for phase, time_ms in phases.items():
        print(f"  {phase:15s}: {time_ms:7.2f} ms ({time_ms/total*100:5.1f}%)")

    return phases
```

**Why this matters:**
- Identifies which phase is slowest
- Focuses subsequent profiling on bottleneck phase
- Uses `record_function` to add custom markers in trace view


**Step 3: Module-Level Profiling**

```python
def profile_model_modules(model, sample_input):
    """Profile time spent in each model module"""

    model.eval()
    sample_input = sample_input.cuda()

    # Add hooks to time each module
    module_times = {}

    def make_hook(name):
        def hook(module, input, output):
            if name not in module_times:
                module_times[name] = []

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            # Forward already happened, this is for next time
            end.record()

            torch.cuda.synchronize()
            # (Note: This is simplified - real implementation more complex)

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # Better approach: Use record_function
    class ProfilingModule(torch.nn.Module):
        def __init__(self, module, name):
            super().__init__()
            self.module = module
            self.name = name

        def forward(self, *args, **kwargs):
            with record_function(f"module_{self.name}"):
                return self.module(*args, **kwargs)

    # Or just use torch.profiler with record_shapes=True
    # It will automatically show module breakdown

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with record_function("model_forward"):
            output = model(sample_input)

    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=20
    ))

    # Clean up
    for hook in hooks:
        hook.remove()
```

**Why this matters:**
- Identifies which model component is slowest
- Guides optimization efforts to specific layers
- Reveals unexpected bottlenecks (e.g., LayerNorm taking 30% of time)


### Phase 4: Identify Operation

**Step 1: Detailed Operation Profiling**

```python
def profile_operations_detailed(model, sample_input):
    """Get detailed breakdown of all operations"""

    model.eval()
    sample_input = sample_input.cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        output = model(sample_input)

    # Group by operation type
    print("\n=== Top Operations by CUDA Time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_src_column_width=100
    ))

    print("\n=== Top Operations by Memory ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=20,
        max_src_column_width=100
    ))

    print("\n=== Grouped by Input Shape ===")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    # Export for trace view
    prof.export_chrome_trace("detailed_trace.json")
    print("\n✅ Exported detailed_trace.json - view in chrome://tracing")

    return prof
```


**Step 2: Reading Profiler Output**

```python
# Example profiler output:
"""
---------------------------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total
---------------------------------  ------------  ------------  ------------  ------------
                      aten::conv2d         0.5%       124ms         45.2%      11.234s
                 aten::convolution         1.2%       298ms         44.7%      11.110s
               aten::_convolution         2.3%       571ms         43.5%      10.812s
          aten::cudnn_convolution        40.1%      9.967s         41.2%      10.241s
                     aten::batch_norm         0.3%        74ms         25.8%       6.412s
                      aten::_batch_norm         1.1%       273ms         25.5%       6.338s
        aten::cudnn_batch_norm        23.2%      5.765s         24.4%       6.065s
                           aten::relu         8.2%      2.038s          8.2%       2.038s
"""
```

**How to interpret:**

| Column | Meaning | When to Look |
|--------|---------|--------------|
| Name | Operation name | Identify what operation |
| Self CPU % | Time in this op only (no children) | Find leaf operations |
| CPU total % | Time in op + children | Find expensive subtrees |
| Self CUDA time | GPU execution time | Main metric for GPU ops |
| Call count | How many times called | High count = optimization target |

**Common patterns:**

```python
# Pattern 1: High aten::copy_ time (40%+)
# → Device transfer issue (CPU ↔ GPU)
# Action: Check device placement, reduce transfers

# Pattern 2: High cudaLaunchKernel overhead
# → Too many small kernel launches
# Action: Increase batch size, fuse operations

# Pattern 3: High cudnn_convolution time
# → Convolutions are bottleneck (expected for CNNs)
# Action: Check input dimensions for Tensor Core alignment

# Pattern 4: High CPU time, low CUDA time
# → CPU bottleneck (data loading, preprocessing)
# Action: Increase num_workers, move ops to GPU

# Pattern 5: Many small operations
# → Operation fusion opportunity
# Action: Use torch.compile or fuse manually
```


**Step 3: Trace View Analysis**

```python
# After exporting trace: prof.export_chrome_trace("trace.json")
# Open in chrome://tracing

"""
Trace view shows:
1. Timeline of GPU kernels
2. CPU → GPU synchronization points
3. Parallel vs sequential execution
4. GPU idle time (gaps between kernels)

What to look for:
- Large gaps between GPU kernels → GPU underutilized
- Many thin bars → Too many small operations
- Thick bars → Few large operations (good for GPU)
- Yellow/red bars → CPU activity (should be minimal during GPU work)
- Overlapping bars → Concurrent execution (good)
"""
```

**Reading trace view:**

```
GPU Stream 0:  ████░░░░████░░░░████  ← Gaps = idle GPU (bad)
GPU Stream 0:  ███████████████████   ← Continuous = good utilization
CPU:           ░░░░████░░░░████░░░░   ← CPU peaks = data loading

Timeline:
[Data Load]──→[GPU Forward]──→[Data Load]──→[GPU Forward]
            ↑ Gap here = GPU waiting for data
```


## Memory Profiling

### Memory Tracking Methodology

**Step 1: Basic Memory Tracking**

```python
import torch

def track_memory(stage_name):
    """Print current memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{stage_name:30s} - Allocated: {allocated:6.2f} GB, "
          f"Reserved: {reserved:6.2f} GB")

# Track at each training phase
track_memory("Start")

data, target = next(iter(dataloader))
data, target = data.cuda(), target.cuda()
track_memory("After data to GPU")

output = model(data)
track_memory("After forward")

loss = criterion(output, target)
track_memory("After loss")

loss.backward()
track_memory("After backward")

optimizer.step()
track_memory("After optimizer step")

optimizer.zero_grad()
track_memory("After zero_grad")
```

**Output interpretation:**

```
Start                          - Allocated:   2.50 GB, Reserved:   2.75 GB
After data to GPU              - Allocated:   2.62 GB, Reserved:   2.75 GB
After forward                  - Allocated:   4.80 GB, Reserved:   5.00 GB  ← Activations
After loss                     - Allocated:   4.81 GB, Reserved:   5.00 GB
After backward                 - Allocated:   7.20 GB, Reserved:   7.50 GB  ← Gradients
After optimizer step           - Allocated:   7.20 GB, Reserved:   7.50 GB
After zero_grad                - Allocated:   4.70 GB, Reserved:   7.50 GB  ← Gradients freed
```

**Key insights:**
- Allocated = actual memory used
- Reserved = memory held by allocator (may be > allocated due to caching)
- Large jump after forward = activations (consider gradient checkpointing)
- Large jump after backward = gradients (same size as parameters)
- Reserved stays high = memory fragmentation or caching


**Step 2: Peak Memory Analysis**

```python
def analyze_peak_memory(model, batch, target):
    """Find peak memory usage and what causes it"""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run one iteration
    data, target = batch.cuda(), target.cuda()

    output = model(data)
    forward_peak = torch.cuda.max_memory_allocated() / 1e9

    loss = criterion(output, target)
    loss_peak = torch.cuda.max_memory_allocated() / 1e9

    loss.backward()
    backward_peak = torch.cuda.max_memory_allocated() / 1e9

    optimizer.step()
    optimizer_peak = torch.cuda.max_memory_allocated() / 1e9

    optimizer.zero_grad()
    final_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Peak after forward:   {forward_peak:.2f} GB")
    print(f"Peak after loss:      {loss_peak:.2f} GB")
    print(f"Peak after backward:  {backward_peak:.2f} GB")
    print(f"Peak after optimizer: {optimizer_peak:.2f} GB")
    print(f"Overall peak:         {final_peak:.2f} GB")

    # Identify bottleneck
    if forward_peak > backward_peak * 0.8:
        print("\n⚠️ Activations dominate memory usage")
        print("   Consider: Gradient checkpointing, smaller batch size")
    elif backward_peak > forward_peak * 1.5:
        print("\n⚠️ Gradients dominate memory usage")
        print("   Consider: Gradient accumulation, mixed precision")
    else:
        print("\n✅ Memory usage balanced across phases")

    return {
        'forward': forward_peak,
        'backward': backward_peak,
        'optimizer': optimizer_peak,
        'peak': final_peak
    }
```


**Step 3: Detailed Memory Summary**

```python
def print_memory_summary():
    """Print detailed memory breakdown"""
    print(torch.cuda.memory_summary())

"""
Example output:
|===========================================================================|
|                  PyTorch CUDA memory summary                              |
|---------------------------------------------------------------------------|
| CUDA OOMs: 0                                                              |
| Metric         | Cur Usage  | Peak Usage | Alloc Retries | # Allocs      |
|----------------|------------|------------|---------------|---------------|
| Allocated      |   4.50 GB  |   7.20 GB  |             0 |       15234   |
| Reserved       |   7.50 GB  |   7.50 GB  |             0 |        1523   |
| Active         |   4.50 GB  |   7.20 GB  |               |               |
| Inactive       |   3.00 GB  |   0.30 GB  |               |               |
|===========================================================================|

Allocated memory:     4.50 GB  ← Actual tensors
Reserved memory:      7.50 GB  ← Memory held by allocator
Active allocations:   4.50 GB  ← Currently in use
Inactive allocations: 3.00 GB  ← Cached for reuse (fragmentation)
"""

# If Inactive >> 0, memory fragmentation is occurring
# Periodic torch.cuda.empty_cache() may help
```

**Fragmentation remediation: `expandable_segments`**

For workloads with variable batch sizes or shape-changing inputs (the most common
fragmentation cause), set the allocator to use expandable segments. This lets a
single CUDA segment grow in place rather than accreting many small unmergeable
allocations:

```bash
# Set BEFORE Python starts (allocator reads env on import torch)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py
```

The environment variable is also accessible as `PYTORCH_ALLOC_CONF` (alias). Per
the CUDA semantics docs, this option is experimental but production-stable for
most training workloads; it specifically helps "jobs changing allocation sizes
frequently, such as having a changing batch size." Cite:
https://docs.pytorch.org/docs/stable/notes/cuda.html

Confirm the fix worked by re-running `print(torch.cuda.memory_summary())` and
checking that `Inactive` drops and `Alloc Retries` stays at 0.


**Step 4: Memory Snapshot (PyTorch 2.0+)**

```python
import pickle
import torch.cuda

def capture_memory_snapshot(filename="memory_snapshot.pickle"):
    """Capture detailed memory snapshot for analysis"""

    # Enable memory history tracking
    torch.cuda.memory._record_memory_history(max_entries=100000)

    try:
        # Run your training code here
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Capture snapshot
        torch.cuda.memory._dump_snapshot(filename)
        print(f"✅ Memory snapshot saved to {filename}")

    finally:
        # Disable tracking
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\nAnalyze with:")
    print(f"  python -m torch.cuda._memory_viz trace_plot {filename}")
    print(f"  # Opens interactive visualization in browser")
```

**Memory snapshot visualization shows:**
- Allocation timeline
- Stack traces for each allocation
- Memory leaks (allocations never freed)
- Fragmentation patterns
- Peak memory events


## GPU Timing Best Practices

### CUDA Synchronization and Events

**❌ WRONG: Using time.time() for GPU operations**

```python
import time

# WRONG - measures CPU time, not GPU time!
start = time.time()
output = model(data)  # Kernel launches, returns immediately
end = time.time()
print(f"Time: {end - start:.4f}s")  # ❌ This is kernel launch overhead (~microseconds)

# Problem: CUDA operations are asynchronous
# time.time() measures CPU time (when kernel was launched)
# Not GPU time (when kernel actually executed)
```

**Why this is wrong:**
- CUDA kernel launches are asynchronous (return immediately to CPU)
- `time.time()` measures CPU wall-clock time
- Actual GPU execution happens later, in parallel with CPU
- Measured time is kernel launch overhead (microseconds), not execution time


**✅ CORRECT: Using CUDA Events**

```python
# CORRECT - measures actual GPU execution time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(data)
end_event.record()

# Wait for GPU to finish
torch.cuda.synchronize()

# Get elapsed time in milliseconds
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"GPU Time: {elapsed_time_ms:.2f} ms")
```

**Why this is correct:**
- CUDA Events are GPU-native timing
- `record()` inserts timing markers into GPU stream
- `synchronize()` waits for GPU to complete
- `elapsed_time()` returns actual GPU execution time


**Alternative: Using torch.profiler**

```python
# For comprehensive profiling, use torch.profiler instead of manual timing
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# This automatically handles synchronization and provides detailed breakdown
```


### Warmup Iterations

**Why warmup is critical:**

```python
# First iteration includes:
# 1. CUDA kernel JIT compilation
# 2. cuDNN algorithm selection (benchmark mode)
# 3. Memory pool allocation
# 4. CPU→GPU transfer of model weights (first time)

# Example timing without warmup:
model.eval()
with torch.no_grad():
    for i in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(sample_input)
        end.record()
        torch.cuda.synchronize()

        print(f"Iteration {i}: {start.elapsed_time(end):.2f} ms")

"""
Output:
Iteration 0: 1234.56 ms  ← JIT compilation, cuDNN benchmarking
Iteration 1:  987.43 ms  ← Still some overhead
Iteration 2:  102.34 ms  ← Stabilized
Iteration 3:  101.89 ms  ← Stable
Iteration 4:  102.12 ms  ← Stable
...
"""
```

**✅ Correct warmup methodology:**

```python
def benchmark_with_warmup(model, sample_input, warmup=5, iterations=100):
    """Proper benchmarking with warmup"""

    model.eval()
    sample_input = sample_input.cuda()

    # Warmup iterations (CRITICAL!)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    # Ensure warmup completed
    torch.cuda.synchronize()

    # Actual measurement
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = model(sample_input)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    # Report statistics
    import numpy as np
    times = np.array(times)

    print(f"Mean:   {times.mean():.2f} ms")
    print(f"Std:    {times.std():.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"Min:    {times.min():.2f} ms")
    print(f"Max:    {times.max():.2f} ms")
    print(f"P95:    {np.percentile(times, 95):.2f} ms")
    print(f"P99:    {np.percentile(times, 99):.2f} ms")

    return times
```

**Warmup rules:**
- Minimum 3 iterations, recommend 5-10
- More complex models need more warmup
- Dynamic control flow needs extra warmup
- Report statistics (mean, std, percentiles), not just average


## Bottleneck Identification Patterns

### CPU-Bound Bottlenecks

**Symptoms:**
- Low GPU utilization (<70%)
- High CPU usage
- Data loading time > computation time
- `nvidia-smi` shows low GPU usage

**Diagnostic code:**

```python
def diagnose_cpu_bottleneck(model, dataloader):
    """Check if training is CPU-bound"""

    # Check GPU utilization
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpu_util = int(result.stdout.strip())

    print(f"GPU Utilization: {gpu_util}%")

    if gpu_util < 70:
        print("⚠️ LOW GPU UTILIZATION - likely CPU-bound")

        # Profile data loading vs compute
        data_time, compute_time = profile_dataloader_vs_model(model, dataloader)

        if data_time > compute_time:
            print("\n🎯 BOTTLENECK: Data loading")
            print("Solutions:")
            print("  1. Increase num_workers in DataLoader")
            print(f"     Current: {dataloader.num_workers}, try: {os.cpu_count()}")
            print("  2. Enable pin_memory=True")
            print("  3. Move data augmentation to GPU (use kornia)")
            print("  4. Cache preprocessed data if dataset is small")
            print("  5. Use faster storage (SSD instead of HDD)")

        else:
            print("\n🎯 BOTTLENECK: CPU preprocessing")
            print("Solutions:")
            print("  1. Move preprocessing to GPU")
            print("  2. Reduce preprocessing complexity")
            print("  3. Batch preprocessing operations")

    else:
        print("✅ GPU utilization is healthy")
        print("   Bottleneck is likely in GPU computation")

    return gpu_util
```

**Common solutions:**

```python
# Solution 1: Increase num_workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Increase from default 0
    pin_memory=True,  # Enable for faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)

# Solution 2: Move augmentation to GPU
import kornia

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.augment = nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.1),
            kornia.augmentation.RandomResizedCrop((224, 224)),
        )

    def forward(self, x):
        return self.augment(x)

# Apply on GPU
gpu_augment = GPUAugmentation().cuda()
for data, target in dataloader:
    data = data.cuda()
    data = gpu_augment(data)  # Augment on GPU
    output = model(data)
```


### GPU-Bound Bottlenecks

**Symptoms:**
- High GPU utilization (>90%)
- Computation time > data loading time
- High CUDA time in profiler

**Diagnostic code:**

```python
def diagnose_gpu_bottleneck(model, sample_input):
    """Identify GPU bottleneck operations"""

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        output = model(sample_input)
        loss = criterion(output, target)
        loss.backward()

    # Find top GPU operations
    events = prof.key_averages()
    cuda_events = [(evt.key, evt.cuda_time_total) for evt in events
                   if evt.cuda_time_total > 0]
    cuda_events.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 GPU operations:")
    total_time = sum(time for _, time in cuda_events)
    for i, (name, time) in enumerate(cuda_events[:10], 1):
        percentage = (time / total_time) * 100
        print(f"{i:2d}. {name:40s} {time/1000:8.2f} ms ({percentage:5.1f}%)")

    # Check for optimization opportunities
    top_op = cuda_events[0][0]
    if 'conv' in top_op.lower():
        print("\n🎯 Bottleneck: Convolution operations")
        print("Solutions:")
        print("  1. Check input dimensions for Tensor Core alignment (multiples of 8)")
        print("  2. Use mixed precision (torch.cuda.amp)")
        print("  3. Consider depthwise separable convolutions")
        print("  4. Profile with different batch sizes")

    elif 'mm' in top_op.lower() or 'matmul' in top_op.lower():
        print("\n🎯 Bottleneck: Matrix multiplication")
        print("Solutions:")
        print("  1. Ensure dimensions are multiples of 8 (FP16) or 16 (BF16)")
        print("  2. Use mixed precision")
        print("  3. Check for unnecessary transposes")

    elif 'copy' in top_op.lower():
        print("\n🎯 Bottleneck: Memory copies")
        print("Solutions:")
        print("  1. Check device placement (CPU ↔ GPU transfers)")
        print("  2. Ensure tensors are contiguous")
        print("  3. Reduce explicit .cuda() or .cpu() calls")

    return cuda_events
```

**Common solutions:**

```python
# Solution 1: Mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(data)
    loss = criterion(output, target)
# 2-3x speedup for large models

# Solution 2: Tensor Core alignment
# Ensure dimensions are multiples of 8 (FP16) or 16 (BF16)
# BAD:  (batch=31, seq_len=127, hidden=509)
# GOOD: (batch=32, seq_len=128, hidden=512)

# Solution 3: torch.compile (PyTorch 2.0+)
model = torch.compile(model)
# Automatic kernel fusion and optimization
```


### Memory-Bound Bottlenecks

**Symptoms:**
- Low GPU utilization despite high memory usage
- Large tensor operations dominating time
- Memory bandwidth saturated

**Diagnostic code:**

```python
def diagnose_memory_bottleneck(model, sample_input):
    """Check if operations are memory-bandwidth limited"""

    # Profile memory and compute
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True
    ) as prof:
        output = model(sample_input)

    # Analyze operations
    for evt in prof.key_averages():
        if evt.cuda_time_total > 0 and evt.self_cuda_memory_usage > 0:
            # Rough FLOP/s estimate
            # Memory-bound: low FLOP/s despite high memory usage
            # Compute-bound: high FLOP/s

            memory_gb = evt.self_cuda_memory_usage / 1e9
            time_s = evt.cuda_time_total / 1e6  # µs to s

            if memory_gb > 1.0 and time_s > 0.01:
                bandwidth = memory_gb / time_s  # GB/s
                print(f"{evt.key:40s}: {bandwidth:.1f} GB/s")

    print("\nIf bandwidth < 500 GB/s, likely memory-bound")
    print("Solutions:")
    print("  1. Reduce intermediate tensor sizes")
    print("  2. Use in-place operations where safe")
    print("  3. Tile large operations")
    print("  4. Increase arithmetic intensity (more compute per byte)")
```


### I/O-Bound Bottlenecks

**Symptoms:**
- Low CPU and GPU utilization
- Long pauses between batches
- Slow disk I/O

**Solutions:**

```python
# Solution 1: Cache dataset in RAM
class CachedDataset(Dataset):
    def __init__(self, dataset):
        self.cache = [dataset[i] for i in range(len(dataset))]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.cache)

# Solution 2: Use SSD storage or RAM disk
# Solution 3: Prefetch data
dataloader = DataLoader(
    dataset,
    num_workers=8,
    prefetch_factor=4,  # Prefetch 4 batches per worker
    pin_memory=True
)
```


## Common Profiling Mistakes

### Mistake 1: Profiling Too Many Iterations

**❌ WRONG:**

```python
# Profiling 100 epochs - output is massive, unusable
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for epoch in range(100):
        for batch in dataloader:
            # ... training ...
            pass
```

**✅ CORRECT:**

```python
# Profile just a few iterations
with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1)
) as prof:
    for epoch in range(1):
        for step, batch in enumerate(dataloader):
            if step >= 10:
                break
            # ... training ...
            prof.step()
```


### Mistake 2: No Warmup Before Timing

**❌ WRONG:**

```python
# Including JIT compilation in timing
start = time.time()
output = model(data)  # First call - includes JIT overhead
end = time.time()
```

**✅ CORRECT:**

```python
# Warmup first
for _ in range(5):
    _ = model(data)

torch.cuda.synchronize()

# Now measure
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(data)
end.record()
torch.cuda.synchronize()
```


### Mistake 3: Synchronization Overhead

**❌ WRONG:**

```python
# Synchronizing in training loop - kills performance!
for batch in dataloader:
    output = model(batch)
    torch.cuda.synchronize()  # ❌ Breaks pipelining!
    loss = criterion(output, target)
    torch.cuda.synchronize()  # ❌ Unnecessary!
    loss.backward()
```

**✅ CORRECT:**

```python
# Only synchronize for timing/profiling, not in production
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    # No synchronization - let GPU pipeline work
```

**When to synchronize:**
- Profiling/timing measurements
- Before memory measurements
- Debugging CUDA errors
- NEVER in production training loop


### Mistake 4: Wrong Profiling Granularity

**❌ WRONG:**

```python
# Profiling entire model - too coarse, can't identify bottleneck
with profile() as prof:
    output = model(data)
# "Model takes 100ms" - not actionable!
```

**✅ CORRECT:**

```python
# Iterative narrowing:
# 1. Profile whole step
# 2. Identify slow phase (forward, backward, optimizer)
# 3. Profile that phase in detail
# 4. Identify specific operation

# Phase 1: Coarse
with record_function("forward"):
    output = model(data)
with record_function("backward"):
    loss.backward()

# Phase 2: Found forward is slow, profile in detail
with profile() as prof:
    output = model(data)
# Now see which layer is slow

# Phase 3: Found layer X is slow, profile that layer
with profile() as prof:
    output = model.layer_x(data)
# Now see which operation in layer X is slow
```


### Mistake 5: Ignoring Memory While Profiling Compute

**❌ WRONG:**

```python
# Only looking at time, ignoring memory
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)
```

**✅ CORRECT:**

```python
# Profile both compute AND memory
with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    output = model(data)

# Check both time and memory
print(prof.key_averages().table(sort_by="cuda_time_total"))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```


### Mistake 6: Profiling in Wrong Mode

**❌ WRONG:**

```python
# Profiling in eval mode when you care about training speed
model.eval()
with torch.no_grad():
    with profile() as prof:
        output = model(data)
# ❌ This doesn't include backward pass!
```

**✅ CORRECT:**

```python
# Profile in the mode you actually use
model.train()
with profile() as prof:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # ✅ Include backward if profiling training
```


## Red Flags - Stop and Profile Systematically

**If you catch yourself thinking ANY of these, STOP and follow methodology:**

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "I can see the bottleneck" | 90% of the time your guess is wrong | Profile to confirm, don't guess |
| "User says X is slow, so X is the bottleneck" | User might be wrong about the cause | Verify with profiling |
| "This loop looks inefficient" | Intuition about performance often wrong | Measure it, don't assume |
| "Profiling takes too long" | Profiling saves hours of guessing | 10 minutes of profiling > hours of guessing |
| "Let me just try this optimization" | Premature optimization wastes time | Measure first, optimize second |
| "It's obviously a GPU problem" | Could be CPU, data loading, or I/O | Check GPU utilization first |
| "I'll reduce batch size" | Doesn't address root cause | Diagnose memory bottleneck first |
| "Skip warmup, it's just one iteration" | First iterations have 10-100x overhead | Always warmup, no exceptions |

**Critical rules:**
1. NEVER optimize before profiling
2. ALWAYS use warmup iterations
3. ALWAYS check GPU utilization before assuming GPU bottleneck
4. ALWAYS profile data loading separately from computation
5. ALWAYS report statistics (mean, std, percentiles), not just average
6. ALWAYS use CUDA Events for GPU timing, never `time.time()`


## Common Rationalizations (Don't Do These)

| Excuse | What Really Happens | Correct Approach |
|--------|-------------------|------------------|
| "User seems rushed, skip profiling" | Guessing wastes MORE time than profiling | 10 min profiling saves hours |
| "I already profiled once" | Might have used wrong tool or granularity | Re-profile with systematic methodology |
| "Profiling overhead will skew results" | Use schedule to minimize overhead | `schedule(wait=1, warmup=2, active=3)` |
| "This worked on another model" | Different models have different bottlenecks | Profile THIS model, not assumptions |
| "Documentation says X is slow" | Depends on context, hardware, data | Verify with profiling on YOUR setup |
| "Just trust the profiler output" | Must interpret correctly | Understand what metrics mean |
| "The model is the bottleneck" | Often it's data loading | Always check data loading vs compute |


## Complete Profiling Example

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule
from torch.utils.data import DataLoader
import numpy as np

def comprehensive_profiling(model, dataloader, device='cuda'):
    """
    Complete profiling workflow following systematic methodology
    """

    print("=" * 80)
    print("PHASE 1: ESTABLISH BASELINE")
    print("=" * 80)

    # Step 1: Measure baseline performance
    model = model.to(device)
    model.train()

    # Warmup
    print("\nWarming up (5 iterations)...")
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Measure baseline
    print("\nMeasuring baseline (10 iterations)...")
    times = []
    for i, (data, target) in enumerate(dataloader):
        if i >= 10:
            break

        data, target = data.to(device), target.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    print(f"\nBaseline Performance:")
    print(f"  Mean:   {times.mean():.2f} ms/iteration")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: IDENTIFY BOTTLENECK TYPE")
    print("=" * 80)

    # Check GPU utilization
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpu_util, mem_used = result.stdout.strip().split(',')
    print(f"\nGPU Utilization: {gpu_util}%")
    print(f"GPU Memory Used: {mem_used} MB")

    if int(gpu_util) < 70:
        print("⚠️ LOW GPU UTILIZATION - likely CPU-bound")
    else:
        print("✅ GPU utilization healthy - likely GPU-bound")

    # Profile data loading vs computation
    print("\nProfiling data loading vs computation...")
    data_times = []
    compute_times = []

    batch_iter = iter(dataloader)
    for i in range(20):
        import time

        # Data loading time
        data_start = time.time()
        data, target = next(batch_iter)
        data_time = time.time() - data_start
        data_times.append(data_time * 1000)  # to ms

        # Computation time
        data, target = data.to(device), target.to(device)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end.record()

        torch.cuda.synchronize()
        compute_times.append(start.elapsed_time(end))

    avg_data = np.mean(data_times)
    avg_compute = np.mean(compute_times)

    print(f"\nData loading: {avg_data:.2f} ms")
    print(f"Computation:  {avg_compute:.2f} ms")

    if avg_data > avg_compute:
        print("🎯 BOTTLENECK: Data loading (CPU-bound)")
    else:
        print("🎯 BOTTLENECK: Model computation (GPU-bound)")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: NARROW TO COMPONENT")
    print("=" * 80)

    # Profile training phases
    print("\nProfiling training phases...")

    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)

    # Forward
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = model(data)
    loss = criterion(output, target)
    end.record()
    torch.cuda.synchronize()
    forward_time = start.elapsed_time(end)

    # Backward
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    backward_time = start.elapsed_time(end)

    # Optimizer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    optimizer.step()
    optimizer.zero_grad()
    end.record()
    torch.cuda.synchronize()
    optimizer_time = start.elapsed_time(end)

    total = forward_time + backward_time + optimizer_time

    print(f"\nPhase breakdown:")
    print(f"  Forward:   {forward_time:7.2f} ms ({forward_time/total*100:5.1f}%)")
    print(f"  Backward:  {backward_time:7.2f} ms ({backward_time/total*100:5.1f}%)")
    print(f"  Optimizer: {optimizer_time:7.2f} ms ({optimizer_time/total*100:5.1f}%)")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 4: IDENTIFY OPERATION")
    print("=" * 80)

    # Detailed profiling
    print("\nRunning detailed profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, (data, target) in enumerate(dataloader):
            if step >= 10:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            prof.step()

    # Print summary
    print("\nTop operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
        max_src_column_width=60
    ))

    print("\nTop operations by memory:")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10,
        max_src_column_width=60
    ))

    prof.export_chrome_trace("detailed_trace.json")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review top operations in table above")
    print("  2. Open chrome://tracing and load detailed_trace.json")
    print("  3. Or view in TensorBoard: tensorboard --logdir=./profiler_logs")
    print("  4. Focus optimization on identified bottleneck")
    print("  5. Re-run this profiling after optimization to verify improvement")

    return {
        'baseline_ms': times.mean(),
        'gpu_utilization': int(gpu_util),
        'data_loading_ms': avg_data,
        'computation_ms': avg_compute,
        'forward_ms': forward_time,
        'backward_ms': backward_time,
        'optimizer_ms': optimizer_time,
    }
```


## Memory Profiling Complete Example

```python
def profile_memory_usage(model, sample_batch, sample_target):
    """Comprehensive memory profiling"""

    print("=" * 80)
    print("MEMORY PROFILING")
    print("=" * 80)

    device = next(model.parameters()).device

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    def print_memory(stage):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{stage:30s} | Alloc: {allocated:5.2f} GB | "
              f"Reserved: {reserved:5.2f} GB | Peak: {peak:5.2f} GB")

    print("\nMemory tracking:")
    print("-" * 80)

    print_memory("Initial")

    # Move data to GPU
    data = sample_batch.to(device)
    target = sample_target.to(device)
    print_memory("After data to GPU")

    # Forward pass
    output = model(data)
    print_memory("After forward")

    # Loss
    loss = criterion(output, target)
    print_memory("After loss")

    # Backward
    loss.backward()
    print_memory("After backward")

    # Optimizer
    optimizer.step()
    print_memory("After optimizer step")

    # Zero grad
    optimizer.zero_grad()
    print_memory("After zero_grad")

    # Final peak
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print("-" * 80)
    print(f"\nPeak memory usage: {peak_memory:.2f} GB")

    # Detailed summary
    print("\n" + "=" * 80)
    print("DETAILED MEMORY SUMMARY")
    print("=" * 80)
    print(torch.cuda.memory_summary())

    # Memory breakdown
    print("\n" + "=" * 80)
    print("MEMORY OPTIMIZATION SUGGESTIONS")
    print("=" * 80)

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    if reserved > allocated * 1.5:
        print("⚠️ Memory fragmentation detected")
        print(f"   Reserved: {reserved:.2f} GB, Allocated: {allocated:.2f} GB")
        print("   Suggestion: Call torch.cuda.empty_cache() periodically")

    # Estimate memory components
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"\nModel parameters: {param_memory:.2f} GB")

    # Estimate gradients (same size as parameters)
    print(f"Gradients (estimate): {param_memory:.2f} GB")

    # Optimizer states (Adam: 2x parameters)
    if isinstance(optimizer, torch.optim.Adam):
        optimizer_memory = param_memory * 2
        print(f"Optimizer states (Adam): {optimizer_memory:.2f} GB")

    # Activations (peak - parameters - gradients - optimizer)
    activation_memory = peak_memory - param_memory - param_memory - optimizer_memory
    if activation_memory > 0:
        print(f"Activations (estimate): {activation_memory:.2f} GB")

        if activation_memory > peak_memory * 0.5:
            print("\n🎯 Activations dominate memory usage")
            print("   Suggestions:")
            print("   1. Use gradient checkpointing")
            print("   2. Reduce batch size")
            print("   3. Use mixed precision (FP16/BF16)")

    return {
        'peak_gb': peak_memory,
        'parameters_gb': param_memory,
        'fragmentation_ratio': reserved / allocated if allocated > 0 else 1.0
    }
```


## Profiling Checklist

Before claiming you've profiled the code, verify:

- [ ] **Baseline established**
  - [ ] Defined performance metric (throughput/latency/memory)
  - [ ] Measured with CUDA Events (not time.time())
  - [ ] Used 5+ warmup iterations
  - [ ] Reported statistics (mean, std, percentiles)
  - [ ] Documented measurement conditions

- [ ] **Bottleneck type identified**
  - [ ] Checked GPU utilization (nvidia-smi)
  - [ ] Profiled data loading vs computation separately
  - [ ] Categorized as CPU-bound, GPU-bound, memory-bound, or I/O-bound
  - [ ] Verified category with profiling data (not guessing)

- [ ] **Component identified**
  - [ ] Profiled training phases (forward/backward/optimizer)
  - [ ] Identified which phase is slowest
  - [ ] Used iterative narrowing approach
  - [ ] Examined both table and trace view

- [ ] **Operation identified**
  - [ ] Profiled bottleneck component in detail
  - [ ] Found specific operation or pattern
  - [ ] Understand WHY it's slow (not just WHAT is slow)
  - [ ] Have actionable optimization target

- [ ] **Verification ready**
  - [ ] Saved baseline measurements
  - [ ] Know how to re-run profiling after optimization
  - [ ] Can verify if optimization actually helped
  - [ ] Have profiling artifacts (traces, summaries)


## Profiling `torch.compile`

`torch.compile` is the production-default kernel-fusion path in PyTorch 2.x and
must be profiled differently from eager mode. The compiled function is traced by
TorchDynamo, optimized by Inductor, and replayed as fused CUDA kernels. Three
phenomena make naive profiling misleading: (1) the first call pays a multi-second
compile cost, (2) input-shape changes can trigger silent recompilation, and
(3) unsupported Python constructs introduce **graph breaks** that fall back to
eager execution and erase most of the speedup.

### Step 1: Measure compile vs eager on the same workload

```python
import torch
import time

def benchmark(fn, *args, n_warmup=10, n_iter=100):
    """Steady-state wall-clock with CUDA sync. See GPU Timing Best Practices."""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter  # ms/iter

eager_model = MyModel().cuda()
compiled_model = torch.compile(eager_model)  # default mode

# WARNING: first call compiles. Discard it from timing.
_ = compiled_model(sample_input)  # warm-up triggers compile

eager_ms = benchmark(eager_model, sample_input)
compiled_ms = benchmark(compiled_model, sample_input)
print(f"Eager:    {eager_ms:.3f} ms/iter")
print(f"Compiled: {compiled_ms:.3f} ms/iter  ({eager_ms / compiled_ms:.2f}x)")
```

**If the speedup is < 1.2x**, you almost certainly have graph breaks or
recompilation. Profile next.

### Step 2: Diagnose graph breaks with `TORCH_LOGS`

Per the torch.compile troubleshooting guide, set environment variables to surface
break and recompile events:

```bash
# Show every graph break with file/line context
TORCH_LOGS="graph_breaks" python train.py

# Show every recompilation and the guard that failed
TORCH_LOGS="recompiles" python train.py

# Both, plus full Dynamo decisions (verbose)
TORCH_LOGS="graph_breaks,recompiles,dynamo" python train.py
```

Each graph break logged means a Python construct Dynamo could not trace; the
compiled region splits and the kernel-fusion benefit ends at that boundary.
Common causes: `.item()` / `.tolist()` calls, data-dependent Python control flow,
`print()` inside the forward, calls into untraceable C extensions.

Cite: https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html

### Step 3: Programmatic inspection with `torch._dynamo.explain`

For a focused break analysis without rerunning the whole training loop:

```python
import torch._dynamo

explanation = torch._dynamo.explain(model)(sample_input)
print(explanation)
# Reports: number of graphs, number of graph breaks, exact reasons,
# and the source line where each break occurred.
```

`torch._dynamo.explain` returns a structured report (graph count, break reasons,
per-break source location) that pinpoints the lines to refactor. Use it as the
first diagnostic when `compiled_ms / eager_ms` is unimpressive.

### Step 4: Deeper Inductor diagnostics

When you need to see the generated kernels themselves (rare; reserve for genuine
optimization work, not first-pass debugging):

```python
import torch._inductor.config
torch._inductor.config.trace.enabled = True   # emit per-graph trace artifacts
# Run the workload; trace files appear under torch_compile_debug/
```

Combine with the eager profile from Step 2 of the systematic methodology to
confirm where compile actually changed the kernel mix.

### Anti-patterns that defeat `torch.compile`

```python
# ANTI-PATTERN 1: tensor-value-dependent control flow
@torch.compile
def forward(x):
    if x.sum() > 0:        # graph break: condition needs .item()
        return x * 2
    return x

# FIX: prefer torch.where / masking
@torch.compile
def forward(x):
    return torch.where(x.sum(keepdim=True) > 0, x * 2, x)

# ANTI-PATTERN 2: shape-dependent Python control flow
@torch.compile
def forward(x):
    if x.size(0) > 32:     # forces recompile per batch size
        return slow_path(x)
    return fast_path(x)
# FIX: use a single path or mark dynamic with torch.compile(dynamic=True)

# ANTI-PATTERN 3: .item() in hot path
loss_value = loss.item()   # syncs CPU/GPU, breaks graph if inside compiled region
# FIX: keep loss as tensor; .item() only at logging boundaries
```

Cross-reference: see `module-design-patterns.md` for compile-friendly module
construction patterns.


## CUDA Graphs

CUDA Graphs eliminate per-kernel launch overhead by capturing a sequence of CUDA
operations once and replaying it as a single submission. They pay off in the
**kernel-launch-bound regime**: small models, very short iterations (typically
sub-millisecond), or large GPU clusters where many small launches dominate the
timeline. Kernel-launch overhead is ~5-10 µs per kernel; if your iteration
contains hundreds of tiny ops, this becomes the bottleneck before any actual
compute does.

CUDA Graphs require **static shapes, static control flow, and a fixed memory
plan** during capture. They do not coexist with `.item()`, dynamic batch sizes,
or per-step Python branches.

### `make_graphed_callables` (recommended for training)

Per the PyTorch reference, `torch.cuda.make_graphed_callables` returns graphed
versions of one or more callables that are autograd-compatible:

```python
torch.cuda.make_graphed_callables(
    callables,                       # Module/Callable or tuple of them
    sample_args,                     # tuple of Tensors, or tuple of tuples
    num_warmup_iters=3,
    allow_unused_input=False,
    pool=None,
)
```

Typical training-loop integration:

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Static shapes required: capture with the same shape used at runtime.
sample_args = (torch.randn(BATCH, FEATURES, device="cuda"),)
graphed_model = torch.cuda.make_graphed_callables(model, sample_args)

for batch in dataloader:
    x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    out = graphed_model(x)              # replays captured graph
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```

Cite: https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html

### Lower-level `torch.cuda.graph` context manager

For custom capture (e.g., capturing a complete forward+backward+optimizer step):

```python
torch.cuda.graph(
    cuda_graph,                      # torch.cuda.CUDAGraph instance
    pool=None,
    stream=None,
    capture_error_mode="global",     # "global" | "thread_local" | "relaxed"
)
```

```python
g = torch.cuda.CUDAGraph()

# Warm up before capture (allocator and cuDNN must settle).
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    out = model(static_input)
    loss = criterion(out, static_target)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()

with torch.cuda.graph(g):
    optimizer.zero_grad(set_to_none=True)
    static_out = model(static_input)
    static_loss = criterion(static_out, static_target)
    static_loss.backward()
    optimizer.step()

# Per-step: copy real data into the captured input buffer, then replay.
for batch in dataloader:
    static_input.copy_(batch[0])
    static_target.copy_(batch[1])
    g.replay()
```

Cite: https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html

### When NOT to use CUDA Graphs

- Iteration time > 5 ms with a handful of large kernels: launch overhead is
  amortized away; capture machinery costs more than it saves.
- Dynamic shapes per step: capture must be redone, eliminating the win.
- Heavy Python work per step (logging, callbacks): not capturable; would need
  refactoring outside the graph.
- Workloads already dominated by a single matmul: graph capture is irrelevant.

If you are not certain you are launch-bound, profile first (Step 1 of the
systematic methodology). The classic signature: high CPU activity in
`cudaLaunchKernel`, low SM occupancy, large gaps between kernels in the trace
view.


## NVTX and Nsight Systems

NVTX (NVIDIA Tools Extension) annotates the GPU timeline with named ranges so a
profiler can group kernels by semantic boundary (e.g., "forward", "block_3",
"attention"). PyTorch's wrapper sits at `torch.cuda.nvtx`. The most useful flow
is to combine NVTX annotations in code with `nsys profile` for capture.

### API surface

```python
# Imperative push/pop (returns nesting depth)
torch.cuda.nvtx.range_push(msg: str) -> int
torch.cuda.nvtx.range_pop() -> int

# Context manager (preferred — exception-safe)
with torch.cuda.nvtx.range("forward"):
    out = model(x)
```

Cite: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html

### Annotation pattern in a training loop

```python
import torch

for step, (x, y) in enumerate(dataloader):
    with torch.cuda.nvtx.range(f"step_{step}"):
        with torch.cuda.nvtx.range("h2d"):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
        with torch.cuda.nvtx.range("forward"):
            out = model(x)
        with torch.cuda.nvtx.range("loss"):
            loss = criterion(out, y)
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        with torch.cuda.nvtx.range("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

### Capturing with Nsight Systems

```bash
# Capture a single training step. Skip the first N steps to avoid compile noise.
nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o train_profile \
    python train.py

# Open the .nsys-rep file in Nsight Systems GUI to see NVTX bands aligned with
# kernels, memcpys, cuDNN calls, and CPU activity.
```

Inside the script, gate capture with `torch.cuda.cudart().cudaProfilerStart()` /
`...Stop()` so only the target steps are recorded. NVTX ranges then become the
labeled lanes in the timeline that make the trace navigable.

For pure-Python timeline annotation that integrates with `torch.profiler` instead
of Nsight, prefer `torch.profiler.record_function("name")`. Use NVTX when you
need the Nsight Systems UI's depth (cuDNN handle reuse, cuBLAS algorithm choice,
SM occupancy alongside the timeline).


## References

**PyTorch Profiling Documentation:**
- torch.profiler: https://docs.pytorch.org/docs/stable/profiler.html
- Profiling recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Performance tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- torch.compile troubleshooting: https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html
- CUDA Graphs: https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html
- make_graphed_callables: https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html
- NVTX: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html
- CUDA memory management (PYTORCH_CUDA_ALLOC_CONF): https://docs.pytorch.org/docs/stable/notes/cuda.html

**Related Skills:**
- tensor-operations-and-memory (memory leak debugging, operation optimization, channels_last)
- mixed-precision-and-optimization (AMP profiling, Tensor Core utilization)
- distributed-training-strategies (multi-GPU profiling)
- module-design-patterns (compile-friendly module construction)
- debugging-techniques (torch.compile debugging deep-dive)

**Cross-pack:**
- yzmir-training-optimization (throughput tuning, batch-size and memory tradeoffs)
- yzmir-llm-specialist (inference optimization, kernel selection)
- yzmir-ml-production (production inference profiling)

**Tools:**
- Chrome tracing: chrome://tracing
- TensorBoard profiler: tensorboard --logdir=<path>
- NVIDIA Nsight Systems: nsys profile python train.py
- PyTorch Memory Visualizer: python -m torch.cuda._memory_viz

---

PyTorch API surface current as of 2026-05 (PyTorch 2.9+); revisit quarterly.
