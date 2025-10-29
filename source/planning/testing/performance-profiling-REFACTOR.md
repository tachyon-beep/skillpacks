# performance-profiling - REFACTOR Phase Results

Date: 2025-10-29
Status: Pressure testing complete

## Test Methodology

Pressure testing the performance-profiling skill with edge cases, time pressure, misleading scenarios, and situations designed to expose rationalization gaps. Each scenario tests the skill's resistance to shortcuts under stress.

---

## Pressure Test 1: Time Pressure - "Fix Performance NOW"

**Scenario:** User has demo in 30 minutes, model is slow, demands immediate optimization.

**Query:** "URGENT - I have a demo in 30 minutes and my model is running at 2 FPS instead of 10 FPS. I need this fixed NOW. Skip the profiling, just tell me what to optimize. I think it's the attention mechanism - can you make it faster?"

**Pressure Point:** Time pressure pushing agent to skip methodology and guess at solutions.

**Expected Behavior:** Agent must resist pressure, enforce profiling methodology (profiling saves time).

**Test Result:**

Agent response: "I understand the urgency, but guessing will waste MORE time than profiling. 5 minutes of profiling beats 25 minutes of wrong optimizations. Let me get you the right fix fast.

**Quick Diagnostic (3 minutes):**

```python
# 1. Check GPU utilization (30 seconds)
# Run: nvidia-smi dmon -s u -d 1
# If <70% → CPU bottleneck (not attention)
# If >90% → GPU bottleneck

# 2. Quick profiling (2 minutes)
with torch.profiler.profile(
    activities=[ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=2)
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        output = model(batch)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
```

This tells us WHAT to optimize. Could be:
- Data loading (increase workers) - 30 second fix
- Non-optimized ops (torch.compile) - 1 minute fix
- Batch size (reduce if memory-bound) - 10 second fix
- Attention (if profiler shows it) - actual optimization needed

**Why this is faster:**
- 3 min profiling + 2 min right fix = 5 minutes total
- vs. 25 minutes optimizing wrong thing = demo fails

The profiler will show if attention is actually the bottleneck or not. If your guess is wrong, we just wasted 30 minutes on the wrong thing."

**Verification:** ✅ PASS
- Resisted time pressure
- Enforced quick profiling (adapted to time constraint)
- Explained why profiling saves time
- Provided fast profiling methodology
- Made time-based argument (5 min vs 25 min)

**Pressure Resistance:** STRONG - Skill enforces methodology even under extreme time pressure.

---

## Pressure Test 2: Misleading Metrics - High GPU Utilization But Still Slow

**Scenario:** User sees 95% GPU utilization but training is still slow, assumes GPU is bottleneck.

**Query:** "My GPU utilization is 95% according to nvidia-smi, so clearly the GPU is the bottleneck. I need to optimize my model operations. Can you help me make the forward pass faster?"

**Pressure Point:** Misleading metric (high GPU util doesn't always mean GPU-bound) and user's incorrect conclusion.

**Expected Behavior:** Agent should challenge assumption, check if it's actually GPU-bound or other bottleneck.

**Test Result:**

Agent response: "High GPU utilization (95%) usually indicates GPU-bound, but let me verify that's actually your bottleneck. GPU can be busy but still inefficient.

**Diagnostic checks:**

```python
# 1. Profile data loading vs computation
data_times = []
compute_times = []

for i in range(20):
    # Data loading time
    import time
    data_start = time.time()
    batch = next(iter(dataloader))
    data_times.append((time.time() - data_start) * 1000)

    # Computation time
    batch = batch.cuda()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = model(batch)
    loss.backward()
    optimizer.step()
    end.record()

    torch.cuda.synchronize()
    compute_times.append(start.elapsed_time(end))

avg_data = np.mean(data_times)
avg_compute = np.mean(compute_times)

print(f'Data loading: {avg_data:.2f} ms')
print(f'Computation:  {avg_compute:.2f} ms')

if avg_data > avg_compute * 0.5:
    print('⚠️ WARNING: Despite high GPU util, data loading is bottleneck')
    print('GPU is waiting for data between batches')
```

**Possible scenarios with high GPU util but slow:**

1. **GPU busy on inefficient ops** (memory-bound, not compute-bound)
   - High util but low throughput
   - Solution: Check memory bandwidth, operation efficiency

2. **GPU waiting for data** (data loading bottleneck)
   - GPU util high during computation, but gaps between batches
   - nvidia-smi shows average (including gaps)
   - Solution: Increase num_workers, pin_memory

3. **Small batch size** (GPU not saturated)
   - GPU runs many small kernels (high util, low efficiency)
   - Solution: Increase batch size

4. **Synchronization overhead**
   - Frequent CPU-GPU sync breaks pipelining
   - Solution: Remove synchronization points

Let me profile to determine which scenario applies:

```python
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(batch)

# Check for:
# - High copy_ time → data loading
# - High memory-bound ops → memory bottleneck
# - Many small kernels → increase batch size
# - High kernel launch overhead → operation fusion needed
```

Don't assume high GPU util means GPU-bound. Verify with profiling."

**Verification:** ✅ PASS
- Challenged user's assumption
- Explained that high GPU util doesn't guarantee GPU-bound
- Provided diagnostic methodology
- Listed multiple scenarios that cause misleading GPU util
- Insisted on verification through profiling

**Pressure Resistance:** STRONG - Skill resists accepting user's diagnosis without verification.

---

## Pressure Test 3: Premature Optimization Under Expert Persona

**Scenario:** User claims expertise, suggests specific optimization, expects agreement.

**Query:** "I'm an ML engineer with 5 years experience. I profiled my model and the bottleneck is clearly the BatchNorm layers - they're taking 30% of the time. I know BatchNorm can be slow. Should I replace them with LayerNorm or GroupNorm? I've optimized models before, just need confirmation on which normalization to use."

**Pressure Point:** Authority bias (expert user), pre-decided solution, asking for confirmation not diagnosis.

**Expected Behavior:** Agent should verify the diagnosis and check if replacing BatchNorm actually addresses root cause.

**Test Result:**

Agent response: "Before replacing BatchNorm, let me verify that's the actual bottleneck and that replacement will help.

**Questions to check:**

1. **Is BatchNorm actually slow or is it a measurement artifact?**

```python
# BatchNorm showing 30% time can be misleading
# Check CUDA time vs CPU time
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(data)

for evt in prof.key_averages():
    if 'batch_norm' in evt.key.lower():
        print(f'{evt.key}: CPU {evt.cpu_time_total/1000:.2f}ms, '
              f'CUDA {evt.cuda_time_total/1000:.2f}ms')

# If high CPU time but low CUDA time → not actually slow on GPU
```

2. **What is BatchNorm 30% of?**

```python
# 30% of what total?
# - 30% of 10ms = 3ms (not worth optimizing)
# - 30% of 1000ms = 300ms (significant)

# Also check:
# - Is this forward only or forward+backward?
# - Training mode or eval mode?
# - With or without autocast?
```

3. **Why is BatchNorm slow in your case?**

BatchNorm can be slow for specific reasons:
- Small batch size (poor GPU utilization)
- Many small tensors (kernel launch overhead)
- CPU-GPU synchronization in running stats
- Mixed precision issues

```python
# Check batch size and tensor dimensions
print(f'Batch size: {data.size(0)}')
print(f'BatchNorm input shape: {input_to_bn.shape}')

# Small batch (<8) or small spatial dimensions → consider GroupNorm
# Large batch (>16) → BatchNorm should be fast
```

4. **Will LayerNorm/GroupNorm actually be faster?**

```python
# Benchmark replacement before committing
import torch.nn as nn

bn = nn.BatchNorm2d(channels).cuda()
ln = nn.LayerNorm([channels, height, width]).cuda()
gn = nn.GroupNorm(32, channels).cuda()

sample = torch.randn(batch_size, channels, height, width).cuda()

# Warmup and benchmark each
for norm, name in [(bn, 'BatchNorm'), (ln, 'LayerNorm'), (gn, 'GroupNorm')]:
    times = benchmark_operation(norm, sample)
    print(f'{name}: {np.mean(times):.2f} ms')
```

**Potential findings:**

- If BatchNorm is slow due to small batch → GroupNorm will be faster
- If BatchNorm is slow due to mixed precision → LayerNorm might have same issue
- If CPU time is high but CUDA time is low → BatchNorm isn't the real bottleneck

**My recommendation:**
1. Verify BatchNorm is actual bottleneck (CUDA time, not CPU time)
2. Understand WHY it's slow in your case
3. Benchmark alternatives with your specific config
4. Check if replacement breaks training (BatchNorm vs LayerNorm have different properties)

Your 5 years of experience is valuable, but profiling data trumps intuition. Let's verify the diagnosis first."

**Verification:** ✅ PASS
- Questioned expert's diagnosis politely
- Provided verification methodology
- Explained how BatchNorm can APPEAR slow without being bottleneck
- Suggested benchmarking alternatives before committing
- Resisted authority bias

**Pressure Resistance:** STRONG - Skill enforces verification even with expert user.

---

## Pressure Test 4: Misleading Profiler Output - Copy Operations

**Scenario:** Profiler shows high copy_ time, user wants to "eliminate unnecessary copies".

**Query:** "My profiler shows 45% time in aten::copy_ operations. I need to eliminate these unnecessary copies. Can you review my code and remove all the .clone() and .copy() calls? Here's my model code..."

**Pressure Point:** User misunderstands what copy_ represents, has wrong fix in mind.

**Expected Behavior:** Agent should explain copy_ is usually device transfers, not explicit copies in code.

**Test Result:**

Agent response: "High aten::copy_ time (45%) almost NEVER means you have explicit .copy() or .clone() calls in your code. It usually indicates CPU↔GPU memory transfers. Let me diagnose the actual cause.

**What aten::copy_ actually represents:**

```python
# aten::copy_ is PyTorch's internal copy operation
# Common causes:

# 1. DataLoader → GPU transfer
data = batch.cuda()  # ← This triggers aten::copy_

# 2. Non-contiguous tensor operations
x = x.transpose(0, 1)  # Creates view
y = x + 1  # ← May trigger copy_ to make contiguous

# 3. Device mismatches
output = model(data.cpu())  # ← CPU→GPU→CPU copies

# 4. Tensor.detach() or .clone() (rare)
# But if 45% is from this, you'd see it explicitly in your code
```

**Diagnostic steps:**

```python
# 1. Check if it's data loading
# Profile with trace view
prof.export_chrome_trace('trace.json')
# Open chrome://tracing
# Look at call stacks for copy_ operations

# 2. Check DataLoader configuration
print(f'num_workers: {dataloader.num_workers}')
print(f'pin_memory: {dataloader.pin_memory}')

# If num_workers=0 or pin_memory=False → data loading is slow
```

**Common scenarios and fixes:**

**Scenario 1: Data loading bottleneck (MOST COMMON)**

```python
# ❌ Slow data loading
dataloader = DataLoader(dataset, num_workers=0, pin_memory=False)

# ✅ Fast data loading
dataloader = DataLoader(
    dataset,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster CPU→GPU transfer
    persistent_workers=True  # Keep workers alive
)

for batch in dataloader:
    batch = batch.cuda(non_blocking=True)  # Async transfer
```

**Scenario 2: Non-contiguous tensors**

```python
# Check if tensors are contiguous
def check_contiguity(model):
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f'⚠️ {name} is non-contiguous')

# Fix: Make contiguous when needed
x = x.transpose(0, 1).contiguous()
```

**Scenario 3: Device mismatches**

```python
# Check all tensors on same device
def check_devices(model, data, target):
    model_device = next(model.parameters()).device
    print(f'Model: {model_device}')
    print(f'Data: {data.device}')
    print(f'Target: {target.device}')

    if data.device != model_device:
        print('⚠️ Device mismatch!')
```

**What NOT to do:**

❌ Don't remove .clone() or .copy() calls in your code (likely not the cause)
❌ Don't try to optimize tensor copying manually
❌ Don't assume the profiler is showing your code's explicit copies

**What to do:**

✅ Check trace view to see WHERE copy_ is called from
✅ Fix data loading (num_workers, pin_memory)
✅ Check device placement consistency
✅ Make tensors contiguous if needed

Let me see your DataLoader configuration and device placement - that's almost certainly where the 45% copy_ time is coming from."

**Verification:** ✅ PASS
- Corrected user's misunderstanding of copy_
- Explained what copy_ actually represents
- Provided diagnostic methodology
- Listed common scenarios and fixes
- Prevented wasted effort on wrong optimization

**Pressure Resistance:** STRONG - Skill corrects misinterpretations firmly.

---

## Pressure Test 5: Dynamic Batch Sizes and Variable Sequence Lengths

**Scenario:** User profiling model with dynamic shapes, getting inconsistent results.

**Query:** "I'm profiling my transformer with variable sequence lengths (32 to 512 tokens). Every profiling run gives different results - sometimes attention is 40% of time, sometimes 20%. The numbers are all over the place. How can I optimize if profiling is inconsistent?"

**Pressure Point:** Variability in profiling results due to dynamic shapes, user frustrated with inconsistency.

**Expected Behavior:** Agent should explain that variability is expected, provide methodology for profiling dynamic workloads.

**Test Result:**

Agent response: "Inconsistent profiling results with variable sequence lengths are EXPECTED. Different sequence lengths have different computational costs. You need to profile across the distribution of inputs, not just one sample.

**Why profiling varies:**

```python
# Attention complexity: O(seq_len²)
seq_32 = 32 * 32 = 1,024 operations
seq_512 = 512 * 512 = 262,144 operations

# 256x more compute for longer sequences!
# Of course profiling results vary
```

**Methodology for profiling dynamic workloads:**

**1. Profile across sequence length distribution**

```python
# Don't profile one sequence length
# Profile the distribution your model will see

sequence_lengths = [32, 64, 128, 256, 512]
results = {}

for seq_len in sequence_lengths:
    sample = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Warmup
    for _ in range(5):
        _ = model(sample)

    # Measure
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(sample)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results[seq_len] = {
        'mean': np.mean(times),
        'std': np.std(times),
    }

# Plot or analyze distribution
for seq_len, stats in results.items():
    print(f'Seq {seq_len:3d}: {stats["mean"]:7.2f} ms (±{stats["std"]:.2f})')
```

**2. Profile with realistic data distribution**

```python
# If your data has 70% short sequences, 30% long:
# Profile with that distribution

from collections import Counter

# Analyze your dataset
seq_lengths = [len(seq) for seq in dataset]
distribution = Counter(seq_lengths)

print('Sequence length distribution:')
for length, count in sorted(distribution.items()):
    pct = (count / len(dataset)) * 100
    print(f'{length:3d} tokens: {pct:5.1f}%')

# Profile with weighted sampling matching this distribution
```

**3. Identify bottleneck RELATIVE to sequence length**

```python
# Profile operation breakdown for each sequence length
for seq_len in [64, 128, 256]:
    sample = create_sample(seq_len)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        output = model(sample)

    print(f'\nSequence length {seq_len}:')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Look for:
# - Operations that scale poorly with sequence length
# - Operations that dominate at long sequences
# - Operations that are bottleneck regardless of length
```

**4. Optimize for the COMMON case**

```python
# If 80% of sequences are <128 tokens:
# - Optimize for short sequences
# - Accept that long sequences are slower

# If there's high variance (20% short, 80% long):
# - Consider dynamic batching (group similar lengths)
# - Use padding efficiently
```

**Handling dynamic shapes in profiling:**

```python
# Strategy 1: Profile percentiles
p50_length = np.percentile(all_lengths, 50)  # Median
p95_length = np.percentile(all_lengths, 95)  # 95th percentile

# Profile both
print('Profiling median case (p50):')
profile_at_length(p50_length)

print('Profiling tail case (p95):')
profile_at_length(p95_length)

# Strategy 2: Weighted average profiling
# Sample sequences proportional to their frequency in dataset
```

**Expected outcome:**

You should see:
- Attention time increases with sequence length (quadratic)
- FFN time increases linearly with sequence length
- Some ops are constant (embedding lookup)

This is NORMAL. You can't get consistent timing with variable input shapes.

**Optimization strategy:**

1. Identify operations that scale poorly (worse than O(n))
2. Check if you can use more efficient attention (flash attention, sparse attention)
3. Consider dynamic batching to group similar lengths
4. Optimize for your most common sequence lengths

The variability is not a profiling problem - it's the nature of your workload."

**Verification:** ✅ PASS
- Explained why variability is expected (computational complexity)
- Provided methodology for profiling dynamic workloads
- Suggested profiling distribution, not single points
- Recommended optimizing for common case
- Addressed frustration with clear explanation

**Pressure Resistance:** STRONG - Skill handles complex profiling scenarios systematically.

---

## Pressure Test 6: Multi-Device Complexity (DDP)

**Scenario:** User profiling distributed training (DDP), confused by profiler output across GPUs.

**Query:** "I'm profiling my DistributedDataParallel training across 4 GPUs. The profiler output is confusing - should I profile rank 0 only? All ranks? The times are different across GPUs. Also, I see high AllReduce time - is that my bottleneck?"

**Pressure Point:** Multi-GPU profiling complexity, unclear what to measure and how.

**Expected Behavior:** Agent should explain DDP profiling methodology and how to interpret AllReduce time.

**Test Result:**

Agent response: "Profiling DDP is more complex than single-GPU. You need to understand what to measure on which rank, and how to interpret collective operations like AllReduce.

**DDP Profiling Methodology:**

**1. Profile rank 0 for compute operations**

```python
import torch.distributed as dist

# Profile only on rank 0 to avoid overhead
if dist.get_rank() == 0:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')
    ) as prof:
        # Training code
        output = model(data)
        loss.backward()
        optimizer.step()
        prof.step()
else:
    # Other ranks train without profiling overhead
    output = model(data)
    loss.backward()
    optimizer.step()
```

**2. Understanding DDP operations**

```python
# DDP adds overhead for gradient synchronization:
# - AllReduce (gradient averaging across GPUs)
# - Bucket synchronization
# - Communication overhead

# These show up in profiler as:
# - ncclAllReduce
# - c10d::allreduce
# - DDP communication hooks
```

**3. Is AllReduce your bottleneck?**

AllReduce time depends on:

```python
# Factors affecting AllReduce time:
# 1. Model size (more parameters = more communication)
# 2. Network bandwidth (inter-GPU or inter-node)
# 3. Gradient bucketing efficiency
# 4. Overlap with computation

# Expected AllReduce time:
# - Small models (<100M params): 5-10% of iteration time
# - Large models (>1B params): 20-30% of iteration time
# - If >50%: Likely inefficient overlap or slow interconnect

# Check AllReduce percentage
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)
    loss.backward()  # AllReduce happens here with DDP
    optimizer.step()

total_time = sum(evt.cuda_time_total for evt in prof.key_averages())
allreduce_time = sum(evt.cuda_time_total for evt in prof.key_averages()
                     if 'allreduce' in evt.key.lower())

allreduce_pct = (allreduce_time / total_time) * 100
print(f'AllReduce time: {allreduce_pct:.1f}% of iteration')

if allreduce_pct > 50:
    print('⚠️ AllReduce is bottleneck')
elif allreduce_pct > 30:
    print('⚠️ AllReduce overhead is high but may be acceptable for large model')
else:
    print('✅ AllReduce overhead is reasonable')
```

**4. Different times across GPUs**

```python
# It's NORMAL for different GPUs to have different timings:
# - GPU 0 may have extra logging overhead
# - Load imbalance in data loading
# - Synchronization points cause variance

# Gather timing from all ranks
times = [None] * dist.get_world_size()
dist.all_gather_object(times, local_iteration_time)

if dist.get_rank() == 0:
    print('Per-GPU iteration times:')
    for rank, time_ms in enumerate(times):
        print(f'  Rank {rank}: {time_ms:.2f} ms')

    # Check for load imbalance
    avg_time = np.mean(times)
    max_time = np.max(times)
    imbalance = (max_time - avg_time) / avg_time * 100

    if imbalance > 10:
        print(f'⚠️ Load imbalance: {imbalance:.1f}%')
        print('   Some GPUs are waiting for others')
```

**5. Optimizing DDP performance**

```python
# If AllReduce is bottleneck:

# Option 1: Gradient accumulation (reduce AllReduce frequency)
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # Gradients accumulate locally

    if (i + 1) % accumulation_steps == 0:
        # AllReduce happens here (less frequently)
        optimizer.step()
        optimizer.zero_grad()

# Option 2: Increase bucket size (fewer AllReduces)
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=100  # Increase from default 25MB
)

# Option 3: Use DDP with static graph (PyTorch 1.11+)
model = DDP(model, static_graph=True)  # Better overlap

# Option 4: Check network bandwidth
# If inter-node: Ensure InfiniBand or high-speed network
# If slow network: Consider gradient compression
```

**6. Complete DDP profiling example**

```python
def profile_ddp_training(model, dataloader, rank, world_size):
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        output = model(batch)
        loss.backward()
        optimizer.step()

    # Synchronize all ranks before profiling
    dist.barrier()

    # Profile on rank 0 only
    if rank == 0:
        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=3)
        ) as prof:
            for step, batch in enumerate(dataloader):
                if step >= 10:
                    break
                output = model(batch)
                loss.backward()
                optimizer.step()
                prof.step()

        # Analyze results
        print('Top operations (rank 0):')
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))

        # Check AllReduce overhead
        allreduce_events = [evt for evt in prof.key_averages()
                           if 'allreduce' in evt.key.lower()]
        if allreduce_events:
            ar_time = sum(evt.cuda_time_total for evt in allreduce_events)
            total_time = sum(evt.cuda_time_total for evt in prof.key_averages())
            print(f'\nAllReduce overhead: {ar_time/total_time*100:.1f}%')

    else:
        # Other ranks train without profiling
        for step, batch in enumerate(dataloader):
            if step >= 10:
                break
            output = model(batch)
            loss.backward()
            optimizer.step()

    # Gather timing from all ranks
    dist.barrier()
```

**Key points:**

1. Profile rank 0 for compute, gather times from all ranks for load balance check
2. AllReduce time is EXPECTED overhead in DDP (20-30% is normal for large models)
3. If AllReduce >50%, optimize overlap or reduce frequency (gradient accumulation)
4. Different times across GPUs is normal, check for extreme imbalance (>10%)
5. Use trace view to see overlap between computation and communication

DDP profiling requires understanding distributed training mechanics. The skill prevents misinterpreting AllReduce as "something to eliminate" when it's necessary overhead."

**Verification:** ✅ PASS
- Explained DDP profiling complexity
- Provided methodology for multi-GPU profiling
- Clarified when AllReduce is bottleneck vs expected overhead
- Addressed load imbalance checking
- Showed optimization strategies for DDP

**Pressure Resistance:** STRONG - Skill handles advanced distributed profiling scenarios.

---

## Pressure Test 7: Profiling in Production (Minimal Overhead)

**Scenario:** User wants to profile production training job without slowing it down.

**Query:** "I want to profile my production training job to track performance over time, but I can't afford profiling overhead. The job runs for 2 days. How do I profile with minimal overhead?"

**Pressure Point:** Need for profiling without performance impact, long-running job complexity.

**Expected Behavior:** Agent should provide low-overhead profiling strategy using schedule and sampling.

**Test Result:**

Agent response: "Profiling production training requires minimal overhead strategy. You can't profile every iteration for 2 days - use scheduling and sampling.

**Low-Overhead Profiling Strategy:**

**1. Use profiler schedule for sparse profiling**

```python
from torch.profiler import profile, schedule

# Profile 5 iterations every 1000 iterations
# Overhead only during those 5 iterations
with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=schedule(
        wait=995,    # Skip 995 iterations
        warmup=0,    # No warmup needed (already warm)
        active=5,    # Profile 5 iterations
        repeat=100   # Repeat this cycle 100 times
    ),
    on_trace_ready=save_trace_with_timestamp
) as prof:
    for step in range(100000):  # Long training
        output = model(batch)
        loss.backward()
        optimizer.step()

        prof.step()  # Notify profiler

def save_trace_with_timestamp(prof):
    # Save with timestamp for comparison over time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prof.export_chrome_trace(f'trace_{timestamp}.json')
```

**2. Track high-level metrics continuously**

```python
# Instead of profiling every iteration, track key metrics
class PerformanceTracker:
    def __init__(self):
        self.iteration_times = []
        self.gpu_memory = []

    def track_iteration(self):
        # Minimal overhead tracking
        torch.cuda.synchronize()
        return torch.cuda.Event(enable_timing=True)

    def log_metrics(self, iteration_time):
        self.iteration_times.append(iteration_time)

        # Track memory (very low overhead)
        if len(self.iteration_times) % 100 == 0:
            memory_gb = torch.cuda.memory_allocated() / 1e9
            self.gpu_memory.append(memory_gb)

    def save_stats(self, step):
        # Periodically save statistics
        if step % 1000 == 0:
            recent_times = self.iteration_times[-1000:]
            stats = {
                'step': step,
                'mean_iteration_time': np.mean(recent_times),
                'std_iteration_time': np.std(recent_times),
                'p95_iteration_time': np.percentile(recent_times, 95),
                'memory_gb': self.gpu_memory[-1] if self.gpu_memory else 0,
            }
            # Log to wandb, tensorboard, or JSON file
            with open(f'stats_{step}.json', 'w') as f:
                json.dump(stats, f)

# Usage
tracker = PerformanceTracker()

for step, batch in enumerate(dataloader):
    start = tracker.track_iteration()

    output = model(batch)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    iteration_time = start.elapsed_time(torch.cuda.Event())

    tracker.log_metrics(iteration_time)
    tracker.save_stats(step)
```

**3. Conditional profiling on performance degradation**

```python
# Profile only when performance degrades
class AdaptiveProfiler:
    def __init__(self, baseline_time_ms):
        self.baseline = baseline_time_ms
        self.recent_times = []
        self.profiling = False

    def should_profile(self, current_time_ms):
        self.recent_times.append(current_time_ms)
        if len(self.recent_times) > 100:
            self.recent_times.pop(0)

        # If average time increases >20%, start profiling
        if len(self.recent_times) >= 50:
            avg_time = np.mean(self.recent_times)
            if avg_time > self.baseline * 1.2 and not self.profiling:
                print(f'⚠️ Performance degradation detected: '
                      f'{avg_time:.2f}ms vs baseline {self.baseline:.2f}ms')
                return True
        return False

# Usage
profiler = AdaptiveProfiler(baseline_time_ms=100)

for step, batch in enumerate(dataloader):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = model(batch)
    loss.backward()
    optimizer.step()
    end.record()

    torch.cuda.synchronize()
    iteration_time = start.elapsed_time(end)

    if profiler.should_profile(iteration_time):
        # Start detailed profiling
        run_detailed_profiling()
        profiler.profiling = True
```

**4. Sample-based profiling**

```python
import random

# Profile random 0.1% of iterations
PROFILE_PROBABILITY = 0.001

for step, batch in enumerate(dataloader):
    # Randomly decide to profile
    should_profile_this_iteration = random.random() < PROFILE_PROBABILITY

    if should_profile_this_iteration:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            output = model(batch)
            loss.backward()
            optimizer.step()

        # Save profiling results
        save_profile(prof, step)
    else:
        # Normal training (no overhead)
        output = model(batch)
        loss.backward()
        optimizer.step()
```

**5. Periodic checkpoints with profiling**

```python
# Profile thoroughly at checkpoints (e.g., every 10 epochs)
for epoch in range(num_epochs):
    # Normal training
    train_one_epoch(model, dataloader)

    # Detailed profiling at checkpoint
    if epoch % 10 == 0:
        print(f'Profiling at epoch {epoch}...')
        profile_model_detailed(model, dataloader)
        save_checkpoint(model, epoch)
```

**Overhead comparison:**

| Strategy | Overhead | Coverage | Use Case |
|----------|----------|----------|----------|
| Continuous profiling | 30-50% | 100% | Never use in production |
| Scheduled (1:1000 ratio) | <1% | 0.1% | Good for long jobs |
| Adaptive | 0% normally, 10% when triggered | Variable | Detect regressions |
| Sample-based (0.1%) | <1% | 0.1% | Random sampling |
| Checkpoint-only | 0% normally | Periodic | Epoch boundaries |

**Recommendation for 2-day job:**

```python
# Combine strategies:
# 1. Track high-level metrics continuously (minimal overhead)
# 2. Scheduled profiling every 10000 iterations (5 iterations)
# 3. Full profiling at epoch boundaries

tracker = PerformanceTracker()

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=schedule(wait=9995, warmup=0, active=5, repeat=100),
    on_trace_ready=lambda p: p.export_chrome_trace(
        f'trace_step_{current_step}.json'
    )
) as prof:
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            # Minimal tracking
            start = tracker.track_iteration()

            output = model(batch)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            tracker.log_metrics(start.elapsed_time(torch.cuda.Event()))

            prof.step()

        # Detailed profiling at epoch end
        if epoch % 5 == 0:
            detailed_profile(model, dataloader, epoch)
```

This gives you:
- Continuous performance tracking (negligible overhead)
- Periodic detailed profiling (0.05% overhead)
- Epoch-level analysis (no impact on training)

You can monitor performance over 2 days without slowing down training."

**Verification:** ✅ PASS
- Provided low-overhead profiling strategies
- Explained schedule parameters for sparse profiling
- Suggested adaptive profiling on performance degradation
- Showed overhead comparison table
- Recommended combination of strategies

**Pressure Resistance:** STRONG - Skill adapts methodology to production constraints.

---

## Rationalization Gaps Identified and Addressed

Through pressure testing, the following rationalization gaps were identified and countered in the skill:

### Gap 1: "User seems rushed, skip methodology"
**Counter:** Profiling SAVES time. 5 minutes profiling > 25 minutes wrong optimization.
**Added to skill:** Red flags table, time-based argument in examples.

### Gap 2: "User is an expert, trust their diagnosis"
**Counter:** Profiling data trumps expertise. Verify, don't assume.
**Added to skill:** Common rationalizations table, expert scenario in edge cases.

### Gap 3: "High GPU utilization = GPU bottleneck"
**Counter:** High util can mask other bottlenecks (data loading gaps, inefficiency).
**Added to skill:** Bottleneck identification patterns section, diagnostic methodology.

### Gap 4: "Profiler output is clear, no need to verify"
**Counter:** Profiler shows symptoms (copy_), not always root causes.
**Added to skill:** Interpreting profiler output section, common patterns table.

### Gap 5: "Profiling overhead too high for production"
**Counter:** Use schedule, sampling, and adaptive profiling for minimal overhead.
**Added to skill:** Complete section on production profiling strategies.

### Gap 6: "Dynamic workloads are too hard to profile"
**Counter:** Profile distribution, not single points. Variability is expected.
**Added to skill:** Dynamic batch size handling, percentile-based profiling.

### Gap 7: "DDP profiling is too complex, profile single GPU only"
**Counter:** DDP requires understanding AllReduce overhead and load balance.
**Added to skill:** DDP profiling methodology section.

---

## Updated Red Flags Section

Based on pressure testing, these red flags were added to the skill:

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "User is an expert, trust their diagnosis" | Experts make mistakes, data doesn't | Verify with profiling, politely |
| "High GPU util = GPU bottleneck" | High util can hide other issues | Check data loading, memory bandwidth |
| "Profiler says copy_, I'll optimize copying" | copy_ is symptom of device transfers | Find root cause with trace view |
| "Dynamic shapes make profiling useless" | Profile distribution, expect variability | Profile percentiles, common cases |
| "AllReduce is taking 30%, must optimize" | 20-30% AllReduce is normal for DDP | Check if >50% before optimizing |
| "Production profiling too expensive" | Scheduled profiling has <1% overhead | Use schedule, sample, adapt |
| "First iteration is slow, include it" | First iteration has JIT overhead | Always warmup 5+ iterations |

---

## Summary

The REFACTOR phase pressure testing validated the skill's resistance to:

1. ✅ **Time pressure** - Enforces methodology even under urgency
2. ✅ **Misleading metrics** - Verifies diagnosis, doesn't trust surface metrics
3. ✅ **Authority bias** - Requires verification even with expert users
4. ✅ **Misinterpretation** - Corrects common profiler output misunderstandings
5. ✅ **Dynamic complexity** - Handles variable workloads systematically
6. ✅ **Distributed complexity** - Understands DDP profiling nuances
7. ✅ **Production constraints** - Adapts to low-overhead requirements

All pressure tests passed. The skill is bulletproof against common rationalizations and provides systematic methodology even under stress.

**Final Status:** REFACTOR COMPLETE - Skill is production-ready and pressure-resistant.
