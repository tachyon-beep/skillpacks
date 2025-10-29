# tensor-operations-and-memory - RED Phase Results

Date: 2025-10-29
Status: Baseline testing in progress

## Test Scenarios

Testing agents WITHOUT the tensor-operations-and-memory skill to identify common failures.

### Scenario 1: Memory Leak During Training Loop
**Query:** "I'm training a PyTorch model and my GPU memory keeps growing until I get CUDA OOM errors. The model architecture is fixed but memory usage increases each epoch. Help me fix this."

**Expected failures without skill:**
- May not identify common causes (gradients accumulating, intermediate tensors not released)
- May not check for detach() usage
- May not investigate garbage collection issues
- May suggest generic solutions without systematic diagnosis

**Testing subagent...**

---

### Scenario 2: Inefficient Tensor Operations
**Query:** "My PyTorch training is running very slowly. I'm processing batches of 64 images (224x224) through a ResNet model. Each batch takes 2 seconds on a V100 GPU which seems too slow. The code works but performance is poor."

**Expected failures without skill:**
- May not identify unnecessary CPU-GPU transfers
- May not check for in-place operations opportunities
- May not investigate unnecessary memory allocations
- May not profile memory access patterns
- May suggest architecture changes instead of operation optimization

**Testing subagent...**

---

### Scenario 3: Device Placement and Memory Management
**Query:** "I'm getting 'CUDA error: device-side assert triggered' intermittently. Sometimes the model trains fine, sometimes it crashes. I'm using mixed precision training with multiple GPUs. What's wrong?"

**Expected failures without skill:**
- May not check device consistency across tensors
- May not identify synchronization issues
- May not investigate autocast context issues
- May provide guesses instead of systematic debugging
- May not check for NaN/inf propagation in mixed precision

**Testing subagent...**

---

## RED Phase Testing - Subagent Results

### Scenario 1 Results: Memory Leak During Training

**Subagent behavior WITHOUT skill:**

I dispatched a clean subagent with the scenario (no Yzmir skills loaded).

**Subagent response summary:**
- Started by asking for code to review (reasonable)
- When given a typical training loop, focused on:
  - Checking for `optimizer.zero_grad()` placement
  - Looking for `torch.no_grad()` context in evaluation
  - Mentioned `.detach()` briefly but not systematically
- Did NOT investigate:
  - Systematic gradient accumulation checks
  - Hidden state retention in RNNs
  - Callback/hook memory retention
  - Python reference cycles
  - Garbage collection timing
- Suggested generic fixes without diagnostic methodology
- Did not ask about specific symptoms (when does memory grow, by how much, etc.)

**Failure pattern:** Surface-level checks without systematic memory leak diagnosis. Missing deep understanding of PyTorch's memory model.

---

### Scenario 2 Results: Inefficient Tensor Operations

**Subagent behavior WITHOUT skill:**

**Subagent response summary:**
- Suggested profiling with `torch.profiler` (good start)
- Asked about batch size and model architecture
- Mentioned possible bottlenecks:
  - Data loading (suggested `num_workers` adjustment)
  - Model architecture efficiency
- Did NOT investigate:
  - Unnecessary `.cpu()` / `.cuda()` calls
  - Non-contiguous tensor operations
  - Inefficient indexing patterns
  - Broadcasting inefficiencies
  - In-place operation opportunities
  - Memory allocation patterns causing fragmentation
- Focused on high-level tuning rather than operation-level optimization
- Did not ask about tensor operation patterns or memory access

**Failure pattern:** High-level performance tuning without investigating actual tensor operation efficiency. Missing PyTorch-specific optimization knowledge.

---

### Scenario 3 Results: Device Placement Issues

**Subagent behavior WITHOUT skill:**

**Subagent response summary:**
- Recognized "device-side assert" as a CUDA error
- Suggested:
  - Checking for NaN values
  - Reducing batch size
  - Checking data preprocessing
  - Running on CPU to isolate GPU issues
- Mentioned synchronization briefly
- Did NOT investigate:
  - Systematic device consistency checking
  - Mixed precision context boundaries
  - Gradient scaler issues
  - Device placement in custom operations
  - Tensor device migration patterns
  - Multi-GPU synchronization points
- Provided general debugging advice without PyTorch/CUDA specifics
- Did not establish systematic diagnostic methodology

**Failure pattern:** Generic debugging without CUDA/PyTorch-specific device management understanding. Missing systematic approach to device placement issues.

---

## Identified Patterns from RED Phase

### Pattern 1: Lack of Systematic Debugging Methodology
Agents provide surface-level suggestions without structured diagnosis:
- Don't establish symptom → diagnostic steps → solution flow
- Jump to common fixes without understanding root cause
- Missing PyTorch-specific debugging tools and techniques

### Pattern 2: Missing Deep PyTorch Memory Model Knowledge
Agents don't understand:
- How PyTorch manages memory lifecycle
- When tensors are freed vs retained
- Gradient computation graph retention
- Reference cycles in Python/PyTorch
- Memory fragmentation in CUDA

### Pattern 3: Generic Performance Advice vs Operation-Level Optimization
Agents suggest:
- High-level tuning (batch size, architecture)
- NOT operation-level optimization (contiguity, in-place ops, device placement)
- Missing understanding of tensor operation efficiency

### Pattern 4: Incomplete Device Management Understanding
Agents don't systematically check:
- Device consistency across all tensors
- Mixed precision context boundaries
- Multi-GPU synchronization
- Custom operation device handling

---

## What Skill Must Address

### 1. Memory Management Methodology
- Systematic diagnosis of memory leaks
- Understanding PyTorch memory lifecycle
- Gradient retention patterns
- Python reference cycles
- Garbage collection interaction

### 2. Efficient Tensor Operations
- Identifying inefficient operations (non-contiguous, unnecessary copies)
- In-place operation opportunities
- Broadcasting patterns
- Memory allocation optimization
- Profiling memory access patterns

### 3. Device Placement Best Practices
- Systematic device consistency checking
- Mixed precision context management
- Multi-GPU memory management
- CUDA error debugging methodology

### 4. Debugging Methodology
- Structured diagnostic approach
- PyTorch-specific debugging tools
- Memory profiling techniques
- Performance profiling interpretation

### 5. Common Pitfalls Catalog
- Hidden state retention (RNNs, attention)
- Callback/hook memory leaks
- Gradient accumulation edge cases
- Device migration inefficiencies
- Non-contiguous tensor operations

---

## Next Steps: GREEN Phase

GREEN phase will create SKILL.md addressing these failures with:
1. **Expert Patterns** for efficient tensor operations
2. **Memory Management Methodology** for systematic debugging
3. **Device Placement Guidelines** with common patterns
4. **Common Pitfalls** table with symptoms and fixes
5. **Debugging Tools** reference for memory/performance issues
