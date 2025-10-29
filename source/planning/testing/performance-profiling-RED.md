# performance-profiling - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Test Methodology

Testing agents WITHOUT the performance-profiling skill to document baseline failures. Each scenario tests profiling methodology gaps that agents commonly exhibit - guessing at bottlenecks, using wrong tools, misinterpreting profiler output, or skipping systematic methodology under pressure.

---

## Scenario 1: Premature Optimization Without Profiling

**Query:** "My PyTorch training is slow - it takes 5 seconds per batch and I need it faster. I think the problem is in my data augmentation pipeline. Can you help me optimize it?"

```python
def train_step(model, batch, optimizer):
    # Data augmentation
    augmented = apply_augmentations(batch['image'])

    # Forward pass
    output = model(augmented)
    loss = criterion(output, batch['label'])

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, batch, optimizer)
        if step % 100 == 0:
            print(f"Loss: {loss:.4f}, Time: {time.time() - start:.2f}s")
```

**Expected Issue:** Agent should refuse to optimize without profiling first. Should establish:
1. Is the bottleneck CPU (data loading) or GPU (computation)?
2. What percentage of time is spent in augmentation vs forward vs backward?
3. Is there a GPU utilization issue (under-utilized GPU means CPU bottleneck)?
4. Need baseline metrics before optimization

**Behavior WITHOUT skill:**

Agent response: "I can help optimize your data augmentation! Here are some improvements:

1. Move augmentations to GPU if possible:
```python
import kornia

def apply_augmentations_gpu(images):
    # Use kornia for GPU-accelerated augmentation
    images = kornia.augmentation.RandomHorizontalFlip()(images)
    images = kornia.augmentation.ColorJitter(0.1, 0.1, 0.1)(images)
    return images
```

2. Use multiple workers in DataLoader:
```python
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

3. Consider using NVIDIA DALI for even faster augmentation.

These should speed up your training significantly!"

**Failure pattern:**
- Agent immediately jumped to solutions without profiling
- Did not verify that augmentation is actually the bottleneck
- Did not ask about GPU utilization or check if GPU is even being used efficiently
- Suggested multiple optimizations that might be irrelevant (if GPU is bottleneck, faster CPU augmentation won't help)
- No measurement methodology to verify improvements
- Violated "measure first, optimize second" principle

**What agent missed:**
- Could be GPU-bound (augmentation speed irrelevant)
- Could be I/O-bound (disk read speed)
- Could be data loader bottleneck (insufficient workers)
- Could be model issue (inefficient forward/backward)
- Without profiling, optimizations are guesses that waste time

---

## Scenario 2: Misinterpreting Profiler Output

**Query:** "I ran torch.profiler on my model and got this output. The top operation is 'aten::copy_' taking 40% of time. Should I optimize my tensor copying?"

```
---------------------------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %
---------------------------------  ------------  ------------  ------------
                      aten::copy_        42.3%      2.134s         42.3%
                 aten::batch_norm        18.5%      934ms          25.2%
                        aten::conv        12.1%      611ms          31.8%
                       aten::addmm         8.2%      414ms           8.2%
                        aten::relu         6.4%      323ms           6.4%
```

**Expected Issue:** Agent should recognize that `aten::copy_` is often symptom, not cause:
1. High copy time usually means CPU↔GPU transfers (device placement issue)
2. Could be non-contiguous tensor operations requiring implicit copies
3. Could be data loading bottleneck (moving data to GPU)
4. Need to check CUDA time, not just CPU time
5. Need to look at trace view to see where copies are called from
6. Should examine with memory profiling to see allocation patterns

**Behavior WITHOUT skill:**

Agent response: "Yes, 40% time in copy operations is significant! Here are ways to reduce copying:

1. Use in-place operations where possible:
```python
# Instead of: x = x + 1
x += 1  # In-place, no copy
```

2. Avoid unnecessary `.clone()` calls in your code
3. Use `.detach()` instead of `.clone().detach()` when you don't need gradients
4. Make sure tensors are contiguous with `.contiguous()`

These should reduce your copy overhead significantly!"

**Failure pattern:**
- Agent focused on reducing copies without understanding WHY copies are happening
- Suggested in-place ops (may break autograd and won't help if copies are from device transfers)
- Did not recognize `aten::copy_` as symptom of deeper issue
- Did not ask to see CUDA profiling (CPU time can be misleading)
- Did not suggest checking where copy_ is being called from (missing stack trace analysis)
- Did not consider that 40% copy time likely means device placement issue, not code issue

**What agent missed:**
- `copy_` often means CPU→GPU transfers (check DataLoader, check device placement)
- Should profile with both CPU and CUDA activities
- Should examine trace view to see call stacks
- High copy time is diagnostic signal, not the problem itself
- Need to identify SOURCE of copies (data loading? model? explicit transfers?)

---

## Scenario 3: Wrong Granularity for Profiling

**Query:** "My model training is slower than expected. I profiled the entire training loop but the output is overwhelming - thousands of operations. How do I find the bottleneck?"

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for epoch in range(10):
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Expected Issue:** Agent should recognize profiling anti-pattern:
1. Profiling too many iterations creates massive output (10 epochs!)
2. Should profile iteratively: whole program → suspect module → specific operation
3. Should use `schedule` to control profiling overhead and focus
4. Should start with coarse profiling (one iteration) then narrow down
5. Should look at both table view and trace view

**Behavior WITHOUT skill:**

Agent response: "The output is overwhelming because you're profiling 10 full epochs! Try profiling just one iteration:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break  # Just one iteration
```

You can also increase `row_limit` to see more operations:
```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
```

This should make the output more manageable!"

**Failure pattern:**
- Agent suggested profiling one iteration (better) but gave no methodology for what to do next
- Did not explain iterative narrowing approach (whole → module → operation)
- Did not suggest using `schedule` for controlled profiling
- Did not mention trace visualization (chrome://tracing) for understanding call stacks
- Did not explain how to interpret results to narrow down to specific bottleneck
- Just reduced data volume, didn't teach profiling methodology

**What agent missed:**
- Systematic narrowing: profile full step, identify slow phase (forward/backward/optimizer), then profile that phase
- Use `record_function` to add custom markers for different model components
- Export trace for visual analysis: `prof.export_chrome_trace("trace.json")`
- Look for patterns: is it one operation repeated many times, or one slow operation?
- Use `with_stack=True` to see call stacks when needed
- Progressive profiling methodology, not just "profile less"

---

## Scenario 4: Ignoring GPU Synchronization in Timing

**Query:** "I'm trying to time different parts of my model to find bottlenecks. I added some timing code but the numbers don't make sense - sometimes negative time differences!"

```python
import time

def profile_model(model, input_data):
    start = time.time()

    # Time forward pass
    forward_start = time.time()
    output = model(input_data)
    forward_time = time.time() - forward_start

    # Time loss computation
    loss_start = time.time()
    loss = criterion(output, target)
    loss_time = time.time() - loss_start

    # Time backward pass
    backward_start = time.time()
    loss.backward()
    backward_time = time.time() - backward_start

    print(f"Forward: {forward_time:.4f}s")
    print(f"Loss: {loss_time:.4f}s")
    print(f"Backward: {backward_time:.4f}s")
    print(f"Total: {time.time() - start:.4f}s")
```

**Expected Issue:** Agent should immediately recognize CUDA asynchrony issue:
1. `time.time()` measures CPU time, not GPU execution time
2. CUDA operations are async - kernel launches return immediately to CPU
3. Must use `torch.cuda.synchronize()` before timing measurements
4. Should use `torch.cuda.Event()` for accurate GPU timing
5. Total time won't equal sum of parts without synchronization

**Behavior WITHOUT skill:**

Agent response: "The issue is that you need to synchronize CUDA operations to get accurate timing:

```python
import time
import torch

def profile_model(model, input_data):
    torch.cuda.synchronize()
    start = time.time()

    output = model(input_data)
    torch.cuda.synchronize()
    forward_time = time.time() - start

    start = time.time()
    loss = criterion(output, target)
    torch.cuda.synchronize()
    loss_time = time.time() - start

    # ... etc
```

Adding `torch.cuda.synchronize()` will make sure GPU operations complete before timing."

**Failure pattern:**
- Agent knew about synchronize() (good!) but provided inefficient implementation
- Did not explain WHY synchronization is needed (CUDA asynchrony)
- Did not mention better approach: `torch.cuda.Event()` for GPU-native timing
- Synchronizing after every operation adds overhead (should use Events instead)
- Did not explain performance cost of synchronization (ruins pipelining)
- Did not suggest using torch.profiler instead of manual timing

**What agent missed:**
- CUDA operations are asynchronous - CPU continues while GPU works
- `time.time()` only measures CPU time (kernel launch overhead, ~microseconds)
- Proper GPU timing uses CUDA Events:
  ```python
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  start_event.record()
  model(input_data)
  end_event.record()
  torch.cuda.synchronize()
  elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
  ```
- Synchronization has performance cost (breaks CUDA stream pipelining)
- For real profiling, use torch.profiler instead of manual timing
- Manual timing with synchronize() is for debugging, not production profiling

---

## Scenario 5: Profiling Without Warmup

**Query:** "I'm profiling my model's inference speed. First iteration takes 2 seconds, but subsequent iterations take 0.1 seconds. Which number is correct?"

```python
model.eval()
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        start = time.time()
        output = model(batch)
        elapsed = time.time() - start
        print(f"Iteration {i}: {elapsed:.4f}s")
        if i >= 10:
            break

# Output:
# Iteration 0: 2.1234s
# Iteration 1: 0.0987s
# Iteration 2: 0.0991s
# Iteration 3: 0.0989s
# ...
```

**Expected Issue:** Agent should explain warmup necessity:
1. First iteration includes CUDA kernel compilation (JIT)
2. First iteration includes memory allocation
3. First iteration includes cuDNN algorithm selection (benchmark mode)
4. Must run warmup iterations before timing
5. Should report statistics (mean, std) over multiple iterations, excluding warmup

**Behavior WITHOUT skill:**

Agent response: "The first iteration is slower because of initialization overhead. You should exclude it from your timing:

```python
model.eval()
times = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        start = time.time()
        output = model(batch)
        elapsed = time.time() - start

        if i > 0:  # Skip first iteration
            times.append(elapsed)

        if i >= 10:
            break

print(f"Average time: {sum(times) / len(times):.4f}s")
```

The subsequent iterations (0.1s) are the correct timing."

**Failure pattern:**
- Agent knew to skip first iteration (partially correct) but didn't explain WHY
- Did not mention CUDA kernel compilation or memory allocation
- Skipped only 1 iteration (should skip 3-5 for full warmup)
- Did not suggest reporting statistics (mean, std, median, percentiles)
- Still using `time.time()` without synchronization (inaccurate for GPU)
- Did not mention warmup is necessary for EVERY profiling methodology

**What agent missed:**
- First iterations include: CUDA kernel JIT compilation, cuDNN benchmarking (if enabled), memory pool allocation
- Typical warmup: 3-5 iterations (not just 1)
- Should report statistics, not just average: mean, std, median, p95, p99
- Warmup applies to torch.profiler too (should skip profiling first few iterations)
- For production benchmarking: warmup + many iterations + statistics
- Still need synchronization for accurate GPU timing
- Warmup is especially critical for models using dynamic control flow or variable shapes

---

## Scenario 6: Missing Memory Profiling for OOM Issues

**Query:** "My model runs fine on batch size 32 but OOMs on batch size 64. I want to optimize it to handle larger batches. How do I profile memory usage?"

**Expected Issue:** Agent should provide memory profiling methodology:
1. Use `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`
2. Use `torch.cuda.memory_summary()` for detailed breakdown
3. Profile memory at different points: after forward, after backward, after optimizer step
4. Use `torch.cuda.memory._record_memory_history()` (PyTorch 2.0+) for detailed tracking
5. Identify which phase uses most memory (activations, gradients, optimizer states)
6. Distinguish between actual memory usage and peak memory usage

**Behavior WITHOUT skill:**

Agent response: "You can check memory usage with:

```python
import torch

allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved: {reserved:.2f} GB")
```

To optimize for larger batches:
1. Use gradient checkpointing to trade compute for memory
2. Use mixed precision training (fp16)
3. Reduce model size or use gradient accumulation

Try these and see if you can fit batch size 64!"

**Failure pattern:**
- Agent mentioned memory functions (good!) but no methodology for WHERE to check
- Jumped to solutions without diagnosing WHAT is using memory
- Did not suggest profiling memory at different stages (forward vs backward vs optimizer)
- Did not mention `torch.cuda.memory_summary()` for detailed breakdown
- Did not explain difference between allocated vs reserved memory
- Suggested gradient checkpointing without knowing if activations are the bottleneck
- No systematic approach to identifying memory bottleneck

**What agent missed:**
- Memory profiling methodology:
  1. Reset peak stats: `torch.cuda.reset_peak_memory_stats()`
  2. Run one iteration
  3. Check peak at each phase: after forward, after backward, after optimizer.step()
  4. Identify which phase has highest memory delta
- Use `torch.cuda.memory_summary()` to see allocation breakdown
- Memory snapshot for detailed analysis: `torch.cuda.memory._record_memory_history()`
- Distinguish between:
  - Activations (reduced by gradient checkpointing)
  - Gradients (same size as parameters)
  - Optimizer states (2x or 3x parameters for Adam)
  - Temporary buffers
- Profile BEFORE suggesting optimizations (gradient checkpointing only helps if activations are bottleneck)

---

## Identified Patterns

### Pattern 1: Guess-First Mentality
Agents jump to optimizations without profiling:
- "This looks slow" → immediate suggestions without measurements
- User says "X is slow" → agent assumes X is the bottleneck without verification
- No baseline metrics collected before optimization attempts
- Solutions proposed based on intuition, not data

### Pattern 2: Tool Misuse
Agents know profiling tools exist but use them incorrectly:
- `time.time()` for GPU operations (ignores CUDA asynchrony)
- Profiling too many or too few iterations (no granularity control)
- Not using warmup iterations (including JIT compilation in timings)
- Not synchronizing before measurements (inaccurate timing)

### Pattern 3: Misinterpretation of Profiler Output
Agents read profiler tables but miss deeper meaning:
- See high `aten::copy_` time → suggest reducing copies (miss that it's symptom of device transfers)
- Look only at CPU time, ignore CUDA time (wrong metric focus)
- Don't examine trace view or call stacks (miss context)
- Focus on individual ops without seeing overall pattern

### Pattern 4: No Systematic Methodology
Agents lack structured approach to profiling:
- No iterative narrowing (whole program → module → operation)
- No comparison of CPU vs GPU vs I/O bound
- No memory profiling alongside compute profiling
- No verification that optimizations actually helped (no before/after measurements)

### Pattern 5: Missing Domain Knowledge
Agents lack deep understanding of:
- CUDA execution model (asynchrony, streams, synchronization cost)
- PyTorch internals (when operations are in FP32 even with autocast, what operations cause copies)
- GPU architecture (memory bandwidth vs compute, Tensor Core requirements)
- Common bottleneck patterns (data loading, memory transfers, inefficient operations)

---

## What Skill Must Address

### 1. Systematic Profiling Methodology
- **Phase 1: Establish baseline**
  - Define performance metric (throughput, latency, memory)
  - Measure end-to-end without profiling overhead
  - Set improvement target
- **Phase 2: Identify bottleneck type**
  - CPU-bound vs GPU-bound vs I/O-bound
  - Check GPU utilization (nvidia-smi)
  - Profile data loading separately from computation
- **Phase 3: Narrow to component**
  - Use torch.profiler at coarse granularity
  - Identify which phase is slow (forward/backward/optimizer/data loading)
  - Focus profiling on that phase
- **Phase 4: Identify operation**
  - Profile suspect component in detail
  - Use trace view to understand call stacks
  - Find specific operation or pattern causing bottleneck

### 2. Tool Selection and Usage
- **When to use what:**
  - `torch.profiler`: Comprehensive profiling (CPU, CUDA, memory)
  - `torch.cuda.Event()`: Accurate GPU timing for specific operations
  - `nvidia-smi`: GPU utilization and memory usage monitoring
  - `torch.cuda.memory_*`: Memory profiling functions
  - `torch.utils.benchmark`: Micro-benchmarking specific operations
- **Correct usage patterns:**
  - Warmup iterations (3-5 before timing)
  - Synchronization (when and why)
  - Profiling granularity (use `schedule` to control overhead)
  - Export and visualization (chrome://tracing for trace view)

### 3. Interpreting Profiler Output
- **Reading tables:**
  - Self CPU vs CPU total vs CUDA time
  - Call counts and averages
  - What high values in specific operations mean
- **Common patterns:**
  - High `aten::copy_` → device placement or contiguity issues
  - High `cudaLaunchKernel` → kernel launch overhead (too many small operations)
  - High CPU time, low GPU time → CPU bottleneck (data loading)
  - Low GPU utilization → underutilized GPU (batch size, architecture)
- **Trace view analysis:**
  - Understanding call stacks
  - Finding sequential vs parallel execution
  - Identifying synchronization points

### 4. Memory Profiling
- **Memory tracking methodology:**
  - Reset peak stats before profiling
  - Measure at each phase (forward, backward, optimizer)
  - Use memory_summary() for breakdowns
  - Memory snapshots for detailed analysis
- **Understanding memory types:**
  - Activations (scales with batch size and depth)
  - Gradients (same size as parameters)
  - Optimizer states (2x-3x parameters)
  - Temporary buffers and fragmentation
- **Memory optimization decision tree:**
  - If activations dominate → gradient checkpointing
  - If optimizer dominates → different optimizer or parameter groups
  - If gradients dominate → gradient accumulation
  - If fragmentation → periodic empty_cache()

### 5. Bottleneck Identification Framework
- **CPU-bound indicators:**
  - Low GPU utilization (<70%)
  - High DataLoader time
  - High CPU time in profiler
  - Solution direction: more workers, faster preprocessing, pin_memory
- **GPU-bound indicators:**
  - High GPU utilization (>90%)
  - High CUDA time in profiler
  - Solution direction: more efficient operations, reduce computation, mixed precision
- **Memory-bound indicators:**
  - Low GPU utilization but high memory usage
  - Large tensor operations with low compute
  - Solution direction: tiling, reduction of intermediate tensors
- **I/O-bound indicators:**
  - Waiting on disk reads
  - Low CPU and GPU utilization
  - Solution direction: faster storage, caching, preprocessing

### 6. Common Profiling Mistakes
Must explicitly counter these mistakes:
- ❌ Using `time.time()` for GPU code (use Events or profiler)
- ❌ Not warming up before timing (includes JIT overhead)
- ❌ Profiling wrong granularity (too broad or too narrow)
- ❌ Optimizing before profiling (premature optimization)
- ❌ Looking only at CPU time (miss GPU bottlenecks)
- ❌ Assuming user's guess is correct (verify with profiling)
- ❌ Not synchronizing before measurements (inaccurate timing)
- ❌ Profiling in wrong mode (train vs eval, with/without autocast)
- ❌ Forgetting about data loading (assume model is bottleneck)
- ❌ Not measuring before and after optimization (no verification)

### 7. Pressure Resistance - Red Flags
Agent must STOP and profile systematically when thinking:
- "I can see the bottleneck just by looking" → NO, profile it
- "User says X is slow, so X is the problem" → Verify with profiling
- "Let me optimize this loop, it looks inefficient" → Measure first
- "Profiling takes too long, I'll guess" → Profiling saves time overall
- "This is obviously a GPU problem" → Check GPU utilization first
- "Just reduce batch size" → Diagnose root cause first

---

## Conclusion

Agents have basic awareness that profiling exists (know about torch.profiler, time measurements) but lack:

1. **Systematic methodology:** No structured approach to narrowing down bottlenecks
2. **Tool expertise:** Incorrect usage (no warmup, wrong timing methods, no synchronization)
3. **Interpretation skills:** Can't read profiler output to identify root causes
4. **Domain knowledge:** Don't understand CUDA asynchrony, GPU architecture, common patterns
5. **Discipline:** Jump to optimization without measurement ("guess-first" instead of "measure-first")

The skill must enforce systematic profiling methodology, teach correct tool usage, provide interpretation frameworks, and build pressure resistance against premature optimization. Agents must learn to MEASURE before OPTIMIZING, every time, no exceptions.
