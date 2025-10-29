# checkpointing-and-reproducibility - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Test Methodology

Testing agents WITHOUT the checkpointing-and-reproducibility skill to document baseline failures. Each scenario tests checkpointing and reproducibility gaps that agents commonly exhibit - incomplete checkpoints (missing optimizer/scheduler/RNG state), incorrect resume logic, false sense of reproducibility from partial seed setting, DDP rank synchronization issues, or checkpoint corruption from improper saving practices.

---

## Scenario 1: Incomplete Checkpoint - Training Diverges on Resume

**Query:** "I'm training a ResNet-50 on ImageNet. I saved a checkpoint at epoch 30 but when I resumed training, the validation loss jumped from 0.45 to 0.52 and never recovered. Can you help me figure out what's wrong?"

```python
# Original training code
import torch
import torch.nn as nn
import torch.optim as optim

model = ResNet50(num_classes=1000).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Save checkpoint
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')

    scheduler.step()
```

```python
# Resume code
checkpoint = torch.load('checkpoint_epoch_30.pt')
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
for epoch in range(start_epoch, 100):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    scheduler.step()
```

**Expected Issue:** Agent should identify that the checkpoint is incomplete:
1. Missing optimizer state_dict (momentum buffers lost, effective reset to SGD without momentum)
2. Missing scheduler state_dict (learning rate resets to 0.1 instead of continuing from 0.01 after step at epoch 30)
3. Missing RNG states (data augmentation randomness not preserved)
4. Should explain WHY this causes divergence: optimizer momentum is critical for SGD convergence
5. Should provide complete checkpoint saving pattern including all state components

**Behavior WITHOUT skill:**

Agent response: "The issue is that you need to also save the optimizer state. Here's the fix:

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
}, f'checkpoint_epoch_{epoch}.pt')

# Resume
checkpoint = torch.load('checkpoint_epoch_30.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

This should fix the divergence issue."

**Failure pattern:**
- Agent identified optimizer state but STILL missed scheduler state (learning rate will reset!)
- Did not mention RNG states for reproducibility
- Did not explain WHY optimizer state matters (momentum buffers)
- Did not calculate what the learning rate SHOULD be at epoch 30 (0.01, not 0.1)
- No validation that checkpoint contains all necessary components
- Did not discuss checkpoint completeness as a general principle

**What agent missed:**
- Scheduler state is equally critical - learning rate schedule continuation
- RNG states for reproducible data augmentation (torch, numpy, python, cuda)
- Should calculate expected LR: epoch 30 means 1 step at epoch 30, so LR = 0.1 * 0.1 = 0.01
- Without scheduler state, optimizer will use CURRENT scheduler (reset to epoch 0), making LR = 0.1
- Checkpoint should include: epoch, model, optimizer, scheduler, loss, rng_state, cuda_rng_state, numpy_rng_state, python_rng_state
- Should validate checkpoint after saving to detect missing components early
- Proper resume means: start_epoch = checkpoint['epoch'] + 1, and scheduler has correct step count

---

## Scenario 2: False Reproducibility - Seeds Not Actually Working

**Query:** "I'm trying to reproduce my experiment results for a paper. I set `torch.manual_seed(42)` at the start but I'm getting different results every time I run training. The losses are close but not identical. What am I doing wrong?"

```python
import torch
import torch.nn as nn
import numpy as np

# Reproducibility setup
torch.manual_seed(42)

# Training
model = SimpleCNN().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for data, target in train_loader:
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Results vary across runs:
# Run 1: Epoch 9, Loss: 0.234567
# Run 2: Epoch 9, Loss: 0.234891
# Run 3: Epoch 9, Loss: 0.234523
```

**Expected Issue:** Agent should identify incomplete seed setting:
1. Only set torch.manual_seed() but not torch.cuda.manual_seed()
2. Missing numpy.random.seed() (if data augmentation uses numpy)
3. Missing python random.seed() (if any Python randomness in data loading)
4. Missing torch.backends.cudnn.deterministic = True (CUDA operations are non-deterministic by default)
5. Missing torch.backends.cudnn.benchmark = False (autotuner introduces randomness)
6. Missing PYTHONHASHSEED environment variable
7. Missing DataLoader worker seed setup (if num_workers > 0)

**Behavior WITHOUT skill:**

Agent response: "You need to also set the CUDA seed. Try this:

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

This should make your results reproducible."

**Failure pattern:**
- Agent added CUDA seed but missed 5+ other sources of randomness
- Did not mention cuDNN deterministic settings (most critical for CUDA reproducibility)
- Did not mention numpy or python random seeds
- Did not mention PYTHONHASHSEED environment variable
- Did not discuss DataLoader worker seeding
- Did not explain WHICH operations are non-deterministic and why
- Did not mention performance tradeoff of deterministic mode

**What agent missed:**
- Complete seed function should set: torch (CPU + CUDA), numpy, python, environment variable
- cuDNN settings are CRITICAL: `torch.backends.cudnn.deterministic = True` and `benchmark = False`
- DataLoader with num_workers > 0 needs worker_init_fn to seed each worker
- Some operations are inherently non-deterministic (atomicAdd, non-deterministic reductions)
- Should link to: https://pytorch.org/docs/stable/notes/randomness.html
- Reproducibility has performance cost (~5-15% slower due to deterministic algorithms)
- Should explain when to prioritize reproducibility (debugging, ablations) vs performance (production training)
- Should test reproducibility by running same code twice and comparing outputs: torch.allclose(out1, out2)

---

## Scenario 3: DDP Checkpoint Corruption - All Ranks Saving

**Query:** "I'm training with DDP across 4 GPUs. My checkpoints are getting corrupted - when I try to load them I get 'RuntimeError: unexpected keys in state_dict' or the file is truncated. Sometimes loading works but the model behavior is wrong. What's happening?"

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP setup
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank}')

model = ResNet50().to(device)
model = DDP(model, device_ids=[rank])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer, criterion)

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
        print(f"Rank {rank}: Saved checkpoint")
```

**Expected Issue:** Agent should identify DDP-specific checkpointing issues:
1. ALL ranks are saving to the same file simultaneously (race condition, file corruption)
2. Should ONLY rank 0 save checkpoints
3. May need to add dist.barrier() before saving to ensure all ranks are synchronized
4. DDP model.state_dict() includes "module." prefix that needs handling
5. Should explain WHY multiple ranks saving causes corruption (concurrent writes, filesystem race)
6. Should mention that ALL ranks can load, but only rank 0 should save

**Behavior WITHOUT skill:**

Agent response: "The issue is that you're saving from all ranks at once. Add a rank check:

```python
if epoch % 10 == 0 and rank == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

This ensures only rank 0 saves the checkpoint."

**Failure pattern:**
- Agent correctly identified rank 0 saving but provided incomplete solution
- Did not add dist.barrier() before or after saving (other ranks should wait)
- Did not mention "module." prefix in DDP state_dict
- Did not explain HOW to load the checkpoint correctly in DDP (all ranks load)
- Did not mention optimizer state in DDP (simple case vs ZeRO optimizer)
- Did not discuss validation after saving (detect corruption early)
- Did not explain WHAT was causing corruption (race condition, concurrent writes)

**What agent missed:**
- Should add dist.barrier() BEFORE saving to ensure all ranks finish training step
- DDP wraps model with "module." prefix, so loading needs: model.load_state_dict(checkpoint['model_state_dict']) when model is already wrapped, or need to handle prefix
- Can use model.module.state_dict() to get unwrapped state_dict for portability
- When loading in DDP, ALL ranks load, but model is already wrapped so each rank initializes correctly
- Should validate checkpoint after saving: if rank == 0: assert os.path.exists(checkpoint_path)
- For ZeRO optimizer (DeepSpeed), optimizer state is sharded across ranks, needs special handling
- Checkpoint location should be on shared filesystem visible to all ranks (not local /tmp)
- Should explain filesystem race conditions: 4 processes writing same file = undefined behavior

---

## Scenario 4: Training Crash Recovery - Loss Jumps After Resume

**Query:** "My training crashed at epoch 47 due to an OOM error. I resumed from my checkpoint at epoch 40, but the loss jumped from 0.31 to 0.48 and training is unstable now. The checkpoint loaded successfully with no errors. Why did this happen?"

```python
# Original training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')

    scheduler.step()

# Resume code
checkpoint = torch.load('checkpoint_epoch_40.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Continue training from epoch 40
for epoch in range(40, 100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    scheduler.step()
```

**Expected Issue:** Agent should identify resume logic errors:
1. Resume loop starts at epoch 40, but checkpoint was saved AFTER epoch 40 finished
2. Should resume from epoch 41, not 40 (starting at checkpoint['epoch'] + 1)
3. Re-running epoch 40 means optimizer takes another step with epoch 40 data, disrupting training trajectory
4. Data loader should also resume from correct position (or use different seed/shuffle)
5. RNG states not saved/restored, so data augmentation randomness differs
6. Should explain the timeline: checkpoint at epoch 40 means "after completing epoch 40", so next epoch is 41

**Behavior WITHOUT skill:**

Agent response: "The issue is that you're starting from epoch 40 instead of epoch 41. Change this:

```python
start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, 100):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    ...
```

This ensures you don't re-run epoch 40."

**Failure pattern:**
- Agent correctly identified the off-by-one error but explanation is incomplete
- Did not explain WHAT happens when you re-run an epoch (optimizer double-steps, trajectory disruption)
- Did not mention RNG state restoration for reproducible data order/augmentation
- Did not discuss data loader position/shuffling considerations
- Did not mention validation after loading (verify model state is sensible)
- Did not explain checkpoint timeline semantics (saved "after" epoch completes)
- Did not discuss best_metric tracking (should also restore best val loss to avoid overwriting)

**What agent missed:**
- Re-running epoch 40 means: (1) optimizer steps on epoch 40 data again, (2) scheduler steps again (LR changes), (3) data augmentation differs without RNG state
- Should save and restore RNG states: torch, cuda, numpy, python
- Should save best_metric (best_val_loss) to avoid checkpoint regression
- Data loader considerations: if shuffling, need same shuffle for epoch 40, or skip to epoch 41
- Should validate model sanity after loading: run on val set, check loss is ~checkpoint['val_loss']
- Checkpoint semantics: "epoch: 40" means "completed epoch 40", so next is 41
- Should save global step counter if using step-based schedules or logging
- Consider saving a "last_checkpoint.pt" that's always most recent, simplifies resume logic

---

## Scenario 5: Checkpoint Version Incompatibility

**Query:** "I trained a model 6 months ago using PyTorch 1.10 and saved checkpoints. Now I'm on PyTorch 2.1 and trying to load my checkpoint, but I get 'AttributeError: Can't get attribute' or some keys are missing. How do I load my old checkpoint?"

```python
# Original training (PyTorch 1.10)
model = TransformerModel(vocab_size=50000, d_model=512, nhead=8)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model_checkpoint.pt')

# Loading in PyTorch 2.1
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# ERROR: some keys missing, unexpected keys, or attribute errors
```

**Expected Issue:** Agent should identify version compatibility issues:
1. PyTorch version changes can affect state_dict keys (architectural changes, new parameters)
2. Pickle protocol changes between versions (some objects not loadable)
3. Should use weights_only=True flag in torch.load() for security (PyTorch 2.0+)
4. Should use strict=False in load_state_dict() to handle missing/extra keys
5. Should log which keys are missing or unexpected to diagnose issue
6. Should explain PyTorch backwards compatibility guarantees (or lack thereof)
7. Should discuss checkpoint migration strategies

**Behavior WITHOUT skill:**

Agent response: "Try loading with strict=False to ignore missing keys:

```python
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

This should allow loading even if some keys don't match."

**Failure pattern:**
- Agent suggested strict=False but didn't explain consequences (missing parameters!)
- Did not mention weights_only flag for security
- Did not suggest logging missing/unexpected keys to understand what changed
- Did not explain WHY keys might be missing (architecture changes in PyTorch internals)
- Did not discuss checkpoint migration strategies (reload in old version, re-save)
- Did not mention that strict=False can silently fail (model partially loaded, may not work correctly)
- Did not discuss checkpoint format versioning best practices

**What agent missed:**
- Should use weights_only=True in torch.load() for security (PyTorch 2.0+): torch.load(path, weights_only=True)
- strict=False is dangerous without verification - model may be partially initialized
- Should log missing keys: missing, unexpected = model.load_state_dict(..., strict=False); print(missing, unexpected)
- Should verify loaded model makes sense: run forward pass, check output shape/values
- Checkpoint migration: load in old PyTorch version, re-save in new version
- Should save PyTorch version in checkpoint metadata: 'pytorch_version': torch.__version__
- Should document checkpoint format and contents in README or checkpoint metadata
- Pickle protocol versioning: torch.save(..., pickle_protocol=4) for compatibility
- Consider using safetensors format for better version stability and security

---

## Scenario 6: Storage Pressure - "Quick Checkpoint" Corners Cut

**Query:** "I'm running a long training job (200 epochs) and disk space is limited. I want to checkpoint efficiently without filling up disk. I'm currently saving every epoch and it's using 500GB. What's the best checkpointing strategy?"

```python
# Current approach - saves every epoch
for epoch in range(200):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Save every epoch (500GB total!)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pt')
```

**Expected Issue:** Agent should provide strategic checkpointing approaches:
1. Multiple checkpoint types: periodic (every N epochs), best (best val loss), last (most recent)
2. Cleanup old checkpoints (keep last N periodic checkpoints)
3. Optimizer state is large (can skip for some checkpoints if only need model for inference)
4. Should explain tradeoffs: fewer checkpoints = less resume granularity, more risk
5. Should mention checkpoint frequency strategies (adaptive based on val loss improvement)
6. Should discuss compression (model state can be compressed)
7. Should mention cloud storage for long-term backups

**Behavior WITHOUT skill:**

Agent response: "You don't need to save every epoch. Save every 10 epochs instead:

```python
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pt')
```

This reduces your storage by 10x."

**Failure pattern:**
- Agent suggested periodic saving but missed other checkpoint types (best, last)
- Did not suggest cleanup policy (disk will still fill up over time)
- Did not mention saving best model separately (most important checkpoint!)
- Did not discuss tradeoffs (10-epoch granularity means losing up to 10 epochs of work)
- Did not mention optimizer state size (can save model-only checkpoints more frequently)
- Did not suggest compression or cloud storage
- Did not provide actual cleanup code (rm old checkpoints)

**What agent missed:**
- Three checkpoint types: (1) best_model.pt (based on val loss), (2) last_checkpoint.pt (always overwrite), (3) checkpoint_epoch_N.pt (periodic, every 10-20 epochs)
- Cleanup policy: keep last N periodic checkpoints, use glob to find and remove old ones
- Model-only checkpoints can be saved more frequently (much smaller, ~50% of full checkpoint)
- Optimizer state includes momentum buffers (often ~2x model size)
- Best model is CRITICAL - always save when val loss improves
- Cleanup code example:
  ```python
  checkpoints = sorted(glob.glob('checkpoint_epoch_*.pt'))
  for old_ckpt in checkpoints[:-keep_last_n]:
      os.remove(old_ckpt)
  ```
- Compression: torch.save(..., _use_new_zipfile_serialization=True) can save 20-30%
- Cloud storage for backups: sync periodic checkpoints to S3/GCS
- Adaptive frequency: save more often when val loss is improving (important training phase)

---

## Baseline Behavior Summary

### Common Patterns Without Skill

**Incomplete checkpoints:**
- Agents typically remember model_state_dict and sometimes optimizer_state_dict
- Consistently forget scheduler_state_dict (learning rate reset on resume!)
- Rarely mention RNG states (torch, cuda, numpy, python)
- Miss additional components: scaler (AMP), best_metric, global_step, config

**False reproducibility:**
- Set torch.manual_seed() but miss torch.cuda.manual_seed()
- Rarely mention cuDNN settings (deterministic, benchmark)
- Don't set numpy or python seeds
- Miss PYTHONHASHSEED environment variable
- Don't understand DataLoader worker seeding (num_workers > 0)

**DDP errors:**
- Understand rank 0 should save but often forget dist.barrier()
- Don't explain "module." prefix handling in DDP state_dict
- Miss that all ranks should load but only rank 0 saves
- Don't discuss shared filesystem requirements
- No mention of ZeRO optimizer state sharding (DeepSpeed)

**Resume logic:**
- Off-by-one errors: starting at checkpoint epoch instead of epoch + 1
- Don't restore RNG states (data order/augmentation differs)
- Don't validate checkpoint after loading (assume it's correct)
- Miss best_metric tracking (overwrite best model on resume)

**Version compatibility:**
- Suggest strict=False without explaining consequences
- Don't mention weights_only flag for security
- Don't log missing/unexpected keys for diagnosis
- No checkpoint migration strategies
- No version metadata in checkpoints

**Storage management:**
- Suggest reducing frequency but miss multi-type strategy (best, last, periodic)
- No cleanup policy (disk fills up over time)
- Don't separate model-only vs full checkpoints
- Miss compression options
- No cloud storage backup strategies

### Knowledge Gaps

Agents consistently lack understanding of:

1. **Checkpoint completeness**: All state components needed for exact resume
2. **Reproducibility multi-library**: Seeds across torch, cuda, numpy, python, environment
3. **DDP synchronization**: Rank 0 saving, barriers, prefix handling
4. **Resume semantics**: Checkpoint at epoch N means "after N", resume at N+1
5. **Validation**: Checking checkpoint integrity, model sanity after loading
6. **Strategic checkpointing**: Multiple types, cleanup, compression, cloud backup
7. **Version compatibility**: Pickle protocols, weights_only, migration strategies
8. **Performance tradeoffs**: Determinism cost, checkpoint I/O overhead

### Rationalization Patterns

**"Just save the model"**
- Agents focus on model as the "important" state
- Rationalize that optimizer will just retrain from checkpoint
- Miss that momentum buffers are critical for convergence
- Don't consider learning rate schedule continuation

**"Random seed is enough"**
- Set one seed and assume reproducibility
- Don't understand cuDNN non-determinism by default
- Miss multi-library seed requirements
- Assume PyTorch handles all randomness internally

**"Checkpointing is simple"**
- Underestimate complexity of complete checkpointing
- Don't validate checkpoints after saving
- Skip strategic checkpoint management
- Assume loading always works correctly

**"Storage pressure justifies shortcuts"**
- Skip optimizer or scheduler to save space
- Don't implement cleanup policy (temporary fix becomes permanent)
- Miss that strategic checkpointing solves storage problem properly

**"DDP is just rank 0 saving"**
- Know to check rank but miss synchronization
- Don't understand state_dict prefix issues
- Assume optimizer state works same as single-GPU

---

## Conclusion

Agents WITHOUT the checkpointing-and-reproducibility skill exhibit systematic failures:

1. **Incomplete checkpoints** lead to training divergence on resume (missing optimizer momentum, wrong learning rate)
2. **False reproducibility** from partial seed setting, missing cuDNN configuration
3. **DDP corruption** from all ranks saving simultaneously, no barriers
4. **Resume logic errors** (off-by-one, missing RNG states) cause trajectory deviation
5. **Version incompatibility** from no migration strategy, dangerous strict=False usage
6. **Storage mismanagement** from no cleanup policy, missing multi-type checkpoint strategy

These failures stem from:
- **Incomplete mental model** of what constitutes a complete checkpoint
- **Underestimating complexity** of reproducibility (multi-library seeds, cuDNN settings)
- **Missing DDP-specific knowledge** (rank synchronization, state_dict prefixes)
- **No validation mindset** (assume checkpoints are correct without checking)
- **Reactive rather than strategic** approach to checkpoint management

The checkpointing-and-reproducibility skill must address these gaps with:
- **Complete checkpoint checklists** (model, optimizer, scheduler, epoch, RNG states, scaler)
- **Full reproducibility setup** (all seeds, cuDNN settings, environment variables)
- **DDP synchronization patterns** (rank 0 saving, barriers, prefix handling)
- **Resume validation** (sanity checks, metric continuity)
- **Strategic checkpoint management** (multiple types, cleanup, compression)
- **Version compatibility** (weights_only, strict=False logging, migration)
- **Rationalization resistance** (tables of "shortcut" vs "consequence")

Testing these scenarios WITHOUT the skill establishes the baseline. The skill must transform these failures into systematic checkpointing practices that ensure training continuity, reproducibility, and checkpoint integrity.
