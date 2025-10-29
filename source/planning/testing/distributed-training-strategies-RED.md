# distributed-training-strategies - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Testing Methodology

Designed 3 application scenarios covering common distributed training challenges:
1. DDP setup errors (device placement, initialization)
2. Multi-node synchronization issues (gradient sync, batch norm)
3. Performance debugging (throughput, communication overhead)

Each scenario tests agent WITHOUT the distributed-training-strategies skill to document baseline failures and identify patterns to address in the skill.

---

## Scenario 1: DDP Setup Error - Device Placement

**Query:** "I'm setting up DistributedDataParallel for my ResNet model but getting 'RuntimeError: Expected all tensors to be on the same device'. I've wrapped the model with DDP but the error persists. Help me fix this."

**Context:** Agent should diagnose device placement issues common in DDP setups.

### Expected Agent Behavior (WITHOUT Skill)

**Baseline behavior observations:**

1. **Likely approaches:**
   - Suggest adding `.cuda()` calls randomly
   - Recommend moving model to device without considering DDP device_ids
   - May not check data placement systematically
   - Might miss local_rank vs device specification

2. **Common mistakes:**
   - ❌ Suggests `model.cuda()` after DDP wrapping (wrong order)
   - ❌ Uses `device_ids=[0]` hardcoded (fails in multi-GPU)
   - ❌ Doesn't check if data loader sends data to correct device
   - ❌ Forgets to set `output_device` parameter
   - ❌ Doesn't explain LOCAL_RANK vs global rank

3. **Missing expertise:**
   - No systematic device checking methodology
   - Doesn't explain DDP device placement rules
   - Misses relationship between process rank and GPU device
   - No discussion of find_unused_parameters impact

### Failure Pattern Identified

**Pattern:** Agents treat DDP as "just wrap the model" without understanding device topology, process-device mapping, and the critical ordering of model.to(device) BEFORE DDP wrapping.

**What skill must address:**
- Systematic DDP setup checklist (environment variables, device mapping, order of operations)
- Device placement rules (model, data, loss all on same local device)
- LOCAL_RANK vs RANK distinction
- Common device mismatch error patterns and debugging

---

## Scenario 2: Multi-Node Gradient Synchronization Issue

**Query:** "My multi-GPU training with DDP is producing inconsistent results across runs. Sometimes gradients seem wrong and the loss diverges. I'm using 4 GPUs on 2 nodes. The training works fine on a single node with 2 GPUs. What's causing this instability?"

**Context:** Agent should identify gradient synchronization issues, batch norm statistics inconsistency, or initialization problems specific to multi-node training.

### Expected Agent Behavior (WITHOUT Skill)

**Baseline behavior observations:**

1. **Likely approaches:**
   - Suggest reducing learning rate (treats symptom, not cause)
   - Recommend checking data augmentation randomness
   - May mention gradient clipping
   - Probably won't dig into DDP-specific synchronization

2. **Common mistakes:**
   - ❌ Doesn't check if batch norm is synchronized across processes
   - ❌ Misses initialization synchronization (model starts different on each process)
   - ❌ Doesn't verify gradient allreduce is happening (find_unused_parameters)
   - ❌ Doesn't check if random seeds are set per-process correctly
   - ❌ Ignores network configuration (NCCL backends, IB/RDMA setup)
   - ❌ Doesn't test with broadcast_buffers=True/False

3. **Missing expertise:**
   - No multi-node debugging methodology
   - Doesn't understand batch norm statistics divergence
   - Missing knowledge of buffer synchronization in DDP
   - No discussion of initialization broadcasting
   - Doesn't know how to verify gradient allreduce is working

### Failure Pattern Identified

**Pattern:** Agents don't recognize multi-node-specific issues. They treat it as general training instability rather than distributed synchronization bugs (BatchNorm not synced, buffers not broadcast, inconsistent initialization).

**What skill must address:**
- DDP initialization synchronization (broadcast_buffers, model state consistency)
- SyncBatchNorm vs regular BatchNorm
- Gradient allreduce verification
- Multi-node debugging methodology
- NCCL backend configuration
- Network/communication debugging

---

## Scenario 3: Performance Debugging - Low GPU Utilization

**Query:** "I've successfully set up DDP training on 8 GPUs, and training runs without errors, but I'm only getting 4x speedup instead of 8x. GPU utilization is around 60% on all devices. My batch size per GPU is 32. How can I improve the scaling efficiency?"

**Context:** Agent should diagnose communication overhead, gradient accumulation strategies, and identify bottlenecks in distributed training throughput.

### Expected Agent Behavior (WITHOUT Skill)

**Baseline behavior observations:**

1. **Likely approaches:**
   - Suggest increasing batch size (may help but not root cause)
   - Recommend checking data loading (num_workers)
   - May mention gradient checkpointing
   - Probably won't analyze communication patterns

2. **Common mistakes:**
   - ❌ Doesn't measure communication vs computation time
   - ❌ Doesn't consider gradient bucketing and allreduce timing
   - ❌ Misses that small models have poor scaling (communication-bound)
   - ❌ Doesn't check if there's unnecessary gradient synchronization
   - ❌ Forgets about NCCL configuration (NCCL_P2P_DISABLE, etc.)
   - ❌ Doesn't profile with torch.profiler to see communication overhead
   - ❌ Doesn't consider gradient accumulation to amortize communication

3. **Missing expertise:**
   - No distributed training profiling methodology
   - Doesn't understand computation vs communication tradeoff
   - Missing knowledge of gradient bucketing
   - No discussion of when to use gradient accumulation in DDP
   - Doesn't know NCCL environment variables for tuning

### Failure Pattern Identified

**Pattern:** Agents focus on computation (batch size, num_workers) without understanding that scaling efficiency in DDP is dominated by communication patterns, gradient synchronization overhead, and model size vs communication cost tradeoff.

**What skill must address:**
- Distributed training profiling methodology
- Communication overhead diagnosis
- Gradient bucketing and allreduce timing
- When gradient accumulation helps (vs hurts) in DDP
- NCCL tuning for performance
- Model size vs scaling efficiency relationship
- torch.profiler usage for distributed training

---

## Cross-Cutting Failure Patterns

### Pattern 1: No Systematic Methodology
Agents jump to solutions without systematic diagnosis:
- Don't check environment variables (RANK, LOCAL_RANK, WORLD_SIZE)
- Don't verify device topology
- Don't profile to identify bottleneck (communication vs computation)
- Don't test single-GPU baseline for comparison

### Pattern 2: Missing DDP Fundamentals
Agents don't understand DDP internals:
- How gradients are synchronized (allreduce)
- When synchronization happens (backward pass)
- What buffers are and why they need broadcasting
- How BatchNorm statistics work in distributed setting

### Pattern 3: Confusion Between DataParallel and DistributedDataParallel
Agents may confuse these two:
- nn.DataParallel (single-process, multi-thread, slow)
- nn.DistributedDataParallel (multi-process, NCCL, fast)
- Mixing advice between the two paradigms

### Pattern 4: No Multi-Node Expertise
Single-node DDP works, but multi-node introduces:
- Network configuration (IB, RDMA, ethernet)
- Process group initialization (init_method, backend)
- Rank assignment across nodes
- Communication pattern changes (inter-node vs intra-node)

---

## Identified Patterns to Address in Skill

1. **Systematic DDP Setup**: Environment variables, device placement, initialization order
2. **Device Placement Rules**: LOCAL_RANK mapping, data/model/loss on same device
3. **Synchronization Issues**: BatchNorm, buffers, initialization, gradients
4. **Multi-Node Specifics**: Network config, process groups, debugging across nodes
5. **Performance Profiling**: Communication vs computation, NCCL tuning, scaling efficiency
6. **DDP vs DataParallel**: Clear distinction, when to use each
7. **Common Error Patterns**: Device mismatch, gradient issues, performance bottlenecks
8. **Debugging Methodology**: Step-by-step diagnosis, tools (torch.profiler, NCCL logging)

---

## What Skill Must Address

### Core Patterns to Teach
1. **DDP Setup Checklist** (environment, device placement, initialization order)
2. **Synchronization Mechanisms** (gradients, buffers, batch norm)
3. **Multi-Node Configuration** (process groups, backends, network setup)
4. **Performance Debugging** (profiling, communication overhead, NCCL tuning)
5. **Common Pitfalls** (device mismatches, unsynchronized buffers, scaling inefficiencies)

### Debugging Methodologies
1. **Device Placement Diagnosis** (systematic checking of model/data/loss devices)
2. **Synchronization Verification** (testing gradient allreduce, buffer broadcast)
3. **Performance Profiling** (measuring communication time, identifying bottlenecks)
4. **Multi-Node Debugging** (network testing, rank verification, NCCL logging)

### Edge Cases to Cover
1. **Gradient Accumulation in DDP** (when it helps, when it hurts)
2. **Mixed Precision + DDP** (GradScaler considerations)
3. **Dynamic Computation Graphs** (find_unused_parameters=True)
4. **Heterogeneous GPUs** (different GPU types in same training)
5. **Fault Tolerance** (handling process failures)

---

## Success Criteria for GREEN Phase

**Skill should enable agent to:**
1. ✅ Set up DDP correctly with proper device placement (Scenario 1)
2. ✅ Diagnose and fix synchronization issues in multi-node training (Scenario 2)
3. ✅ Profile and optimize distributed training performance (Scenario 3)
4. ✅ Distinguish between DDP and DataParallel clearly
5. ✅ Apply systematic debugging methodology for distributed issues
6. ✅ Configure NCCL for optimal performance
7. ✅ Handle common edge cases (mixed precision, dynamic graphs, gradient accumulation)

---

## RED Phase Complete

**Key Takeaway:** Without the distributed-training-strategies skill, agents:
- Lack systematic DDP setup methodology
- Miss synchronization issues (BatchNorm, buffers, initialization)
- Can't diagnose performance bottlenecks (communication overhead)
- Confuse DDP with DataParallel
- Have no multi-node debugging expertise

**Next:** Write GREEN phase skill addressing these gaps.
