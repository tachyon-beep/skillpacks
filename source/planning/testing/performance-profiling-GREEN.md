# performance-profiling - GREEN Phase Results

Date: 2025-10-29
Status: Skill verification complete

## Test Methodology

Testing agents WITH the performance-profiling skill to verify it addresses all baseline failures identified in RED phase. Each scenario from RED phase is retested with the skill active.

---

## Scenario 1: Premature Optimization Without Profiling

**Query:** "My PyTorch training is slow - it takes 5 seconds per batch and I need it faster. I think the problem is in my data augmentation pipeline. Can you help me optimize it?"

**Expected Behavior WITH Skill:**
Agent should refuse to optimize without profiling first, establish baseline, identify bottleneck type, then provide targeted solutions.

**Agent Response WITH Skill:**

"Before optimizing, we need to profile systematically to identify the actual bottleneck. Your guess might be wrong - let me measure first.

**Phase 1: Establish Baseline**

Let's measure current performance properly:

```python
# Baseline measurement with warmup
def measure_baseline(model, dataloader):
    # Warmup (critical - don't skip!)
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        train_step(model, batch, optimizer)

    # Measure
    times = []
    for i, batch in enumerate(dataloader):
        if i >= 20:
            break

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        train_step(model, batch, optimizer)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    times = np.array(times)
    print(f'Baseline: {times.mean():.2f} ms/batch (std: {times.std():.2f})')
    return times.mean()
```

**Phase 2: Identify Bottleneck Type**

Now let's check if it's actually data augmentation:

```python
# Check GPU utilization
# Run: nvidia-smi dmon -s u -d 1

# Profile data loading vs computation
def separate_data_vs_compute(model, dataloader):
    data_times = []
    compute_times = []

    batch_iter = iter(dataloader)
    for i in range(20):
        # Time data loading
        import time
        data_start = time.time()
        batch = next(batch_iter)
        data_times.append((time.time() - data_start) * 1000)

        # Time computation
        torch.cuda.synchronize()
        comp_start = torch.cuda.Event(enable_timing=True)
        comp_end = torch.cuda.Event(enable_timing=True)

        comp_start.record()
        train_step(model, batch, optimizer)
        comp_end.record()

        torch.cuda.synchronize()
        compute_times.append(comp_start.elapsed_time(comp_end))

    avg_data = np.mean(data_times)
    avg_compute = np.mean(compute_times)

    print(f'Data loading: {avg_data:.2f} ms')
    print(f'Computation:  {avg_compute:.2f} ms')

    if avg_data > avg_compute:
        print('BOTTLENECK: Data loading')
    else:
        print('BOTTLENECK: Model computation')
```

Based on profiling results, I'll provide targeted recommendations. If data loading IS the bottleneck, then yes, we optimize augmentation. If GPU is the bottleneck, augmentation speed is irrelevant."

**Verification:** ✅ PASS
- Agent refused to optimize without measurement
- Enforced systematic methodology
- Would measure before suggesting solutions
- No premature optimization

---

## Scenario 2: Misinterpreting Profiler Output

**Query:** "I ran torch.profiler and got this output. The top operation is 'aten::copy_' taking 40% of time. Should I optimize my tensor copying?"

**Expected Behavior WITH Skill:**
Agent should recognize `aten::copy_` as symptom of device placement or data loading issues, not a target for optimization.

**Agent Response WITH Skill:**

"High `aten::copy_` time (40%) is a SYMPTOM, not the root cause. This almost always indicates CPU↔GPU memory transfers, not a code issue you can optimize directly.

**Diagnostic Steps:**

1. **Check if it's data loading:**
```python
# Profile with both CPU and CUDA activities
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in dataloader:
        batch = batch.cuda()  # ← This causes copy_
        output = model(batch)
        break

# Check CUDA time vs CPU time
print(prof.key_averages().table(sort_by='cuda_time_total'))
```

If `copy_` has high CPU time but low CUDA time → it's data transfer from DataLoader.

2. **Check device placement:**
```python
# Are tensors moving between devices unnecessarily?
print(f'Model device: {next(model.parameters()).device}')
for name, param in model.named_parameters():
    if param.device.type != 'cuda':
        print(f'⚠️ {name} is on {param.device}')
```

3. **Look at trace view:**
Export the trace and view in chrome://tracing:
```python
prof.export_chrome_trace('trace.json')
```

Look for where `copy_` is called from (stack traces). It's likely:
- DataLoader → GPU transfer (solve with pin_memory=True, non_blocking=True)
- Non-contiguous tensors (solve with .contiguous())
- Explicit .cuda() calls in hot loop (solve with better device management)

**Solutions based on root cause:**

If data loading:
```python
dataloader = DataLoader(
    dataset,
    pin_memory=True,  # Faster CPU→GPU transfer
    num_workers=4
)

for batch in dataloader:
    batch = batch.cuda(non_blocking=True)  # Async transfer
```

If non-contiguous:
```python
# Check and fix
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
```

**Don't waste time on:**
- ❌ Using in-place ops (won't help device transfers)
- ❌ Reducing .clone() calls (copy_ is device transfer, not cloning)
- ❌ Manual tensor optimization (root cause is elsewhere)

The skill enforces: diagnose WHY copy_ is high, fix root cause, don't treat symptom."

**Verification:** ✅ PASS
- Recognized copy_ as symptom
- Provided diagnostic methodology
- Suggested checking trace view and call stacks
- Directed to root cause (device transfers)
- Prevented wasted optimization effort

---

## Scenario 3: Wrong Granularity for Profiling

**Query:** "My model training is slower than expected. I profiled the entire training loop but the output is overwhelming - thousands of operations. How do I find the bottleneck?"

**Expected Behavior WITH Skill:**
Agent should teach iterative narrowing methodology and proper profiling granularity control.

**Agent Response WITH Skill:**

"You profiled at the wrong granularity. The skill teaches iterative narrowing: whole program → suspect phase → specific operation. Let me guide you:

**Step 1: Coarse Profiling - Identify Phase**

```python
from torch.profiler import record_function

# Don't profile everything - profile phases
data, target = next(iter(dataloader))
data, target = data.cuda(), target.cuda()

# Time each phase
with record_function('forward_pass'):
    output = model(data)
    loss = criterion(output, target)

with record_function('backward_pass'):
    loss.backward()

with record_function('optimizer_step'):
    optimizer.step()
    optimizer.zero_grad()
```

Run profiler on this, you'll see which phase is slowest.

**Step 2: Control Profiling Scope**

Use schedule to avoid overwhelming output:

```python
from torch.profiler import profile, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=1,    # Skip first iteration
        warmup=2,  # Warmup iterations (no profiling)
        active=3,  # Profile these 3 iterations only
        repeat=1   # Do this once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
        # training code
        prof.step()  # Notify profiler
```

This profiles only 3 iterations, not everything.

**Step 3: Narrow to Component**

Say forward pass is slow. Profile just forward:

```python
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
```

Now you see top 20 operations in forward pass only.

**Step 4: Use Trace View for Context**

```python
prof.export_chrome_trace('trace.json')
# Open chrome://tracing, load trace.json
```

Trace view shows timeline, call stacks, helps understand where time is spent.

**Systematic Narrowing:**
1. Profile whole training step → find slowest phase
2. Profile that phase → find slowest component
3. Profile that component → find slowest operation
4. Now you have actionable target

Don't profile 10 epochs. Don't profile thousands of operations at once. Narrow iteratively."

**Verification:** ✅ PASS
- Taught iterative narrowing methodology
- Explained schedule for controlling overhead
- Showed how to use record_function for phase markers
- Mentioned trace view for visualization
- Prevented profiling too much at once

---

## Scenario 4: Ignoring GPU Synchronization in Timing

**Query:** "I'm trying to time different parts of my model. I added timing code but the numbers don't make sense - sometimes negative time differences!"

**Expected Behavior WITH Skill:**
Agent should immediately identify CUDA asynchrony issue and provide correct GPU timing methodology.

**Agent Response WITH Skill:**

"The issue is CUDA asynchrony. You're using `time.time()` which measures CPU time, not GPU execution time. CUDA kernel launches are asynchronous - they return to CPU immediately.

**Why your timing is wrong:**

```python
import time
start = time.time()
output = model(data)  # Kernel launches, returns to CPU immediately
end = time.time()
# ❌ This measures kernel launch overhead (~microseconds), not execution time!
```

CUDA operations are async. The GPU is still working when `time.time()` measures.

**✅ Correct GPU Timing: Use CUDA Events**

```python
# CUDA Events measure actual GPU execution time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(data)
end_event.record()

torch.cuda.synchronize()  # Wait for GPU to finish
elapsed_ms = start_event.elapsed_time(end_event)  # Actual GPU time

print(f'GPU Time: {elapsed_ms:.2f} ms')
```

**Why this is correct:**
- Events are GPU-native timing markers
- `record()` inserts marker in GPU stream
- `synchronize()` waits for completion
- `elapsed_time()` returns actual GPU execution time

**For comprehensive profiling, use torch.profiler:**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)

print(prof.key_averages().table(sort_by='cuda_time_total'))
```

This automatically handles synchronization and provides detailed breakdown.

**⚠️ Warning: Don't synchronize in production training loop!**

```python
# ❌ WRONG - kills performance
for batch in dataloader:
    torch.cuda.synchronize()  # Breaks GPU pipelining!
    output = model(batch)

# ✅ CORRECT - only synchronize for profiling
for batch in dataloader:
    output = model(batch)  # Let GPU pipeline work
```

Synchronization has performance cost. Use for profiling/debugging, never in production loop."

**Verification:** ✅ PASS
- Immediately identified CUDA asynchrony issue
- Explained WHY time.time() is wrong
- Provided correct CUDA Events methodology
- Warned about synchronization overhead
- Suggested torch.profiler as better alternative

---

## Scenario 5: Profiling Without Warmup

**Query:** "I'm profiling my model's inference speed. First iteration takes 2 seconds, but subsequent iterations take 0.1 seconds. Which number is correct?"

**Expected Behavior WITH Skill:**
Agent should explain warmup necessity and provide correct benchmarking methodology.

**Agent Response WITH Skill:**

"The first iteration includes CUDA kernel JIT compilation, cuDNN benchmarking, and memory allocation. You need warmup iterations to exclude this overhead.

**Why first iteration is slow:**

1. **CUDA kernel JIT compilation** - kernels compiled on first use
2. **cuDNN algorithm selection** - if `cudnn.benchmark=True`, cuDNN tests algorithms
3. **Memory pool allocation** - first allocation reserves memory
4. **CPU→GPU transfers** - initial data movement

These happen once, not every iteration. They're setup overhead, not actual model speed.

**✅ Correct Benchmarking with Warmup:**

```python
model.eval()
sample_input = sample_input.cuda()

# Warmup (CRITICAL - don't skip!)
print('Warming up...')
with torch.no_grad():
    for _ in range(5):  # Minimum 3, recommend 5-10
        _ = model(sample_input)

torch.cuda.synchronize()  # Ensure warmup completed

# Now measure
times = []
with torch.no_grad():
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(sample_input)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

# Report statistics (not just average!)
import numpy as np
times = np.array(times)
print(f'Mean:   {times.mean():.2f} ms')
print(f'Std:    {times.std():.2f} ms')
print(f'Median: {np.median(times):.2f} ms')
print(f'P95:    {np.percentile(times, 95):.2f} ms')
print(f'P99:    {np.percentile(times, 99):.2f} ms')
```

**How many warmup iterations?**
- Minimum: 3
- Recommended: 5-10
- Complex models or dynamic control flow: 10+

**Report statistics, not just average:**
- Mean (average latency)
- Std (consistency)
- Median (typical case)
- P95, P99 (tail latency - important for production)

**The 0.1s timing is correct** (after warmup). The 2s includes setup overhead that only happens once.

**Warmup applies to ALL profiling:**
- torch.profiler (use schedule with warmup)
- Manual timing
- Benchmarking different implementations

Never include first iteration in measurements."

**Verification:** ✅ PASS
- Explained what happens in first iteration
- Provided correct warmup methodology
- Specified warmup iteration count (5-10)
- Emphasized reporting statistics, not just average
- Clarified that 0.1s is correct timing

---

## Scenario 6: Missing Memory Profiling for OOM Issues

**Query:** "My model runs fine on batch size 32 but OOMs on batch size 64. How do I profile memory usage to optimize?"

**Expected Behavior WITH Skill:**
Agent should provide systematic memory profiling methodology to identify what's using memory.

**Agent Response WITH Skill:**

"Before optimizing for larger batches, we need to profile memory to identify WHAT is using it. Then we can apply targeted solutions.

**Step 1: Track Memory at Each Phase**

```python
def track_memory(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f'{stage:25s} - Alloc: {allocated:5.2f} GB, Reserved: {reserved:5.2f} GB')

torch.cuda.reset_peak_memory_stats()

track_memory('Start')

data, target = batch.cuda(), target.cuda()
track_memory('After data to GPU')

output = model(data)
track_memory('After forward')

loss = criterion(output, target)
track_memory('After loss')

loss.backward()
track_memory('After backward')

optimizer.step()
track_memory('After optimizer step')

optimizer.zero_grad()
track_memory('After zero_grad')

peak = torch.cuda.max_memory_allocated() / 1e9
print(f'\nPeak memory: {peak:.2f} GB')
```

**Step 2: Identify Memory Bottleneck**

```python
# If large jump after forward → activations dominate
# If large jump after backward → gradients dominate
# If high after optimizer → optimizer states dominate

# Example output:
# After forward:  4.8 GB  ← Big jump (2.3 GB) = activations
# After backward: 7.2 GB  ← Jump (2.4 GB) = gradients
# After optimizer: 7.2 GB ← No change = optimizer states were already allocated

if forward_memory_jump > 2.0:
    print('BOTTLENECK: Activations')
    print('Solution: Gradient checkpointing')
```

**Step 3: Detailed Memory Summary**

```python
print(torch.cuda.memory_summary())
# Shows:
# - Allocated vs Reserved (fragmentation)
# - Active vs Inactive allocations
# - Number of allocs/frees
```

**Step 4: Targeted Solutions Based on Bottleneck**

```python
# If ACTIVATIONS dominate:
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Trade compute for memory
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# If GRADIENTS dominate:
# Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# If OPTIMIZER STATES dominate:
# Use different optimizer or parameter groups
optimizer = torch.optim.SGD(...)  # Instead of Adam (3x less memory)
```

**Step 5: Verify Improvement**

After applying solution, re-run memory profiling:
```python
# Before: Peak 8.2 GB (OOM at batch 64)
# After gradient checkpointing: Peak 5.1 GB (fits batch 64)
```

Don't guess at solutions. Profile memory, identify bottleneck, apply targeted fix, verify."

**Verification:** ✅ PASS
- Systematic memory tracking methodology
- Identifies WHICH phase uses memory
- Provides diagnostic interpretation
- Suggests targeted solutions based on bottleneck
- Emphasizes verification after optimization

---

## Summary of Improvements

### Before Skill (RED Phase)
Agents exhibited these failures:
1. ❌ Jumped to optimization without profiling
2. ❌ Misinterpreted profiler output (treated symptoms)
3. ❌ Profiled at wrong granularity (overwhelming output)
4. ❌ Used time.time() for GPU code (wrong tool)
5. ❌ Skipped warmup iterations (included JIT overhead)
6. ❌ No memory profiling methodology (guessed at solutions)

### After Skill (GREEN Phase)
Agents now demonstrate:
1. ✅ Refuse to optimize without measurement first
2. ✅ Interpret profiler output correctly (find root causes)
3. ✅ Use iterative narrowing (systematic granularity)
4. ✅ Use CUDA Events for GPU timing (correct tool)
5. ✅ Always warmup before timing (exclude JIT overhead)
6. ✅ Systematic memory profiling (identify actual bottleneck)

### Key Behavioral Changes

**Methodology Enforcement:**
- Agents now follow Four-Phase Framework: Baseline → Bottleneck Type → Component → Operation
- No premature optimization - measurement always comes first
- Iterative narrowing instead of profiling everything at once

**Tool Mastery:**
- Correct tool selection: torch.profiler for comprehensive, CUDA Events for specific timing
- Proper tool usage: warmup, synchronization, schedule for overhead control
- Understanding tool output: CPU vs CUDA time, call counts, memory usage

**Interpretation Skills:**
- Recognize symptoms vs root causes (copy_ → device transfers)
- Read profiler tables and trace views correctly
- Understand what metrics actually mean

**Domain Knowledge Applied:**
- CUDA asynchrony and synchronization
- Warmup necessity (JIT, cuDNN, allocation)
- Memory components (activations, gradients, optimizer states)
- Bottleneck categories (CPU-bound, GPU-bound, memory-bound, I/O-bound)

---

## Skill Validation

The performance-profiling skill successfully addresses all baseline failures:

| RED Scenario | Failure Pattern | GREEN Result | Status |
|--------------|----------------|--------------|--------|
| Scenario 1 | Premature optimization | Enforces measurement first | ✅ FIXED |
| Scenario 2 | Misinterpret profiler output | Identifies root causes | ✅ FIXED |
| Scenario 3 | Wrong granularity | Teaches iterative narrowing | ✅ FIXED |
| Scenario 4 | Wrong timing tool | Uses CUDA Events | ✅ FIXED |
| Scenario 5 | No warmup | Always warmup first | ✅ FIXED |
| Scenario 6 | No memory methodology | Systematic memory profiling | ✅ FIXED |

**Conclusion:** The skill successfully transforms agent behavior from "guess and optimize" to "measure, identify, fix, verify". All RED phase failures are addressed.
