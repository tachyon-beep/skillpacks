# debugging-techniques - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Test Methodology

Testing agents WITHOUT the debugging-techniques skill to document baseline failures. Each scenario tests debugging methodology gaps that agents commonly exhibit - random trial-and-error, print-driven debugging, no systematic hypothesis formation, misunderstanding PyTorch-specific debugging tools, or skipping root cause analysis under pressure.

---

## Scenario 1: Cryptic Shape Mismatch Error

**Query:** "My PyTorch model is crashing with this error. Can you help me fix it?"

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = MyModel()
x = torch.randn(4, 3, 32, 32)
output = model(x)
```

**Error message:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
```

**Expected Issue:** Agent should use systematic debugging methodology:
1. Read the FULL error message (4x57600 suggests flatten produced 57600 features, but fc1 expects 64)
2. Work backwards: fc1 input is 57600, but defined as expecting 64
3. Calculate what conv1 output shape is: (4, 64, 30, 30) → flatten → 57600 features
4. Realize the mismatch: fc1 expects 64 features but gets 57600
5. Fix: change fc1 input size to match actual flattened size OR use adaptive pooling

**Behavior WITHOUT skill:**

Agent response: "The error is a shape mismatch in your linear layer. Let me fix it:

```python
def forward(self, x):
    x = self.conv1(x)
    x = x.view(x.size(0), 64)  # Reshape to match fc1 input
    x = self.fc1(x)
    x = self.fc2(x)
    return x
```

Try this - I'm reshaping to match the linear layer's expected input size."

**Failure pattern:**
- Agent did not read or analyze the error message properly (4x57600 tells you the actual shape!)
- Suggested arbitrary reshape without understanding WHERE the 57600 comes from
- Did not calculate conv output shape (30x30x64 = 57600)
- Did not explain WHY there's a mismatch (conv output size not matching fc1 expectation)
- Suggested reshape that will FAIL (can't reshape 57600 to 64 without losing data)
- No systematic debugging - just guessed at a "fix"

**What agent missed:**
- Error message contains critical diagnostic info: "4x57600" tells you actual tensor shape
- Should work backwards from error to source
- Should add shape assertions to verify understanding: `assert x.shape[1] == 64`
- Should calculate conv output size: (H-K+1) * (W-K+1) * C = (32-3+1) * (32-3+1) * 64 = 57600
- Proper fix requires either: (1) change fc1 input size to 57600, or (2) add pooling to reduce spatial dimensions
- Random reshaping will corrupt data or fail

---

## Scenario 2: Silent Failure - Model Not Learning

**Query:** "My model isn't learning - the loss stays constant at around 2.3 for 10 epochs. I checked and it's definitely training. What's wrong?"

```python
model = SimpleClassifier(input_dim=784, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Expected Issue:** Agent should systematically diagnose silent failures:
1. Recognize pattern: constant loss ~2.3 for 10 classes = log(10) = random guessing
2. Form hypothesis: model outputs not changing (stuck weights) OR optimizer not updating weights
3. Test hypothesis systematically:
   - Check if gradients exist: `for name, param in model.named_parameters(): print(name, param.grad)`
   - Check if gradients are non-zero
   - Check if weights are changing between epochs
   - Check model output distribution (all same? all zeros?)
4. Notice missing `optimizer.step()` call
5. Verify fix by checking loss decreases after adding step()

**Behavior WITHOUT skill:**

Agent response: "If your model isn't learning, try these common fixes:

1. Increase learning rate:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 10x higher
```

2. Check your loss function is appropriate for the task
3. Make sure your data is normalized:
```python
data = (data - data.mean()) / data.std()
```

4. Try a different optimizer:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Try these and see if training improves!"

**Failure pattern:**
- Agent jumped straight to random solutions without diagnosing the problem
- Did not recognize the diagnostic signal: loss ~2.3 = log(10) = random predictions
- Did not form hypothesis about WHAT could cause constant loss
- Did not suggest checking gradients or weight updates
- Suggested changes that are irrelevant (higher LR won't help if optimizer.step() is missing)
- No systematic investigation - just threw common fixes at the wall

**What agent missed:**
- **Diagnostic signal:** Constant loss at log(num_classes) means model outputting uniform random predictions
- **Systematic diagnosis:**
  1. Check gradients exist and are non-zero
  2. Check weights are updating between steps
  3. Check model outputs are varying (not stuck)
  4. Check optimizer.step() is called
- **Root cause:** Missing `optimizer.step()` means gradients computed but never applied
- Should verify hypothesis with targeted checks before suggesting fixes
- After identifying missing step(), should verify fix works (loss decreases)

---

## Scenario 3: Intermittent Error - Works Sometimes

**Query:** "I'm getting this strange error that only happens sometimes - maybe 1 in 10 runs. When it happens, training crashes at a random batch with 'RuntimeError: CUDA error: device-side assert triggered'. What's causing this?"

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.fc(x)

model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=5)
criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    input_ids, labels = batch
    output = model(input_ids)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Expected Issue:** Agent should recognize intermittent error debugging pattern:
1. "device-side assert" usually means invalid index access (embedding lookup out of range)
2. Intermittent suggests data-dependent issue (some batches have invalid data)
3. Hypothesis: some input_ids >= vocab_size or < 0
4. Test hypothesis:
   - Add assertion before embedding: `assert (input_ids >= 0).all() and (input_ids < 10000).all()`
   - Or add bounds checking: `print(f"Min: {input_ids.min()}, Max: {input_ids.max()}")`
5. Find which batches have out-of-range indices
6. Fix data preprocessing to ensure valid indices

**Behavior WITHOUT skill:**

Agent response: "CUDA errors can be tricky! The 'device-side assert' error often appears cryptically. Here are steps to debug:

1. Move your model to CPU to get a clearer error:
```python
model = model.cpu()
# Run training on CPU
```

2. Add `.cuda()` explicitly to make sure tensors are on correct device:
```python
input_ids = input_ids.cuda()
labels = labels.cuda()
```

3. Try catching the error:
```python
try:
    output = model(input_ids)
    loss = criterion(output, labels)
    loss.backward()
except RuntimeError as e:
    print(f"Error at batch: {batch}")
    print(f"Input shape: {input_ids.shape}")
    continue
```

Run on CPU first to see a clearer error message!"

**Failure pattern:**
- Agent recognized CUDA errors are cryptic (good!) but didn't identify the SPECIFIC pattern
- Suggested moving to CPU (good general advice) but no hypothesis about WHAT to look for
- Did not recognize "device-side assert" specifically indicates invalid index access
- Did not suggest checking for out-of-bounds indices in data
- Try/except will hide the error, not fix it
- No systematic approach to reproducing or diagnosing intermittent errors

**What agent missed:**
- **Pattern recognition:** "device-side assert" + Embedding = out-of-bounds index (this is a known pattern)
- **Intermittent error methodology:**
  1. Make error deterministic (fix random seed, identify triggering batch)
  2. Minimize reproduction (isolate the failing batch)
  3. Inspect data that causes failure
- **Hypothesis:** If 1 in 10 batches fail, likely some data has invalid indices (>= vocab_size or negative)
- **Verification:** Add assertions before embedding to catch bad data: `assert input_ids.max() < vocab_size`
- **Root cause:** Data preprocessing or tokenization producing invalid indices
- Moving to CPU reveals: `IndexError: index 10000 is out of bounds for dimension 0 with size 10000`

---

## Scenario 4: NaN Loss After Several Epochs

**Query:** "Training was going fine - loss was decreasing normally for 5 epochs. Then suddenly at epoch 6, the loss became NaN and stayed NaN. What happened?"

```python
class MultiLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = MultiLayerMLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        data, target = batch
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Avg Loss: {total_loss / len(train_loader):.4f}")
```

**Expected Issue:** Agent should use PyTorch's NaN debugging tools:
1. Recognize pattern: NaN appearing after N epochs suggests gradients or activations exploding
2. Use `torch.autograd.set_detect_anomaly(True)` to pinpoint where NaN originates
3. Run with anomaly detection to find exact operation producing NaN
4. Inspect gradients and activations before NaN appears (forward/backward hooks)
5. Likely causes: high learning rate (0.1 is high), no gradient clipping, numerical instability
6. Verify: check gradient magnitudes before NaN epoch, add gradient clipping

**Behavior WITHOUT skill:**

Agent response: "NaN loss usually means your gradients exploded. Here are common fixes:

1. Reduce learning rate:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Much lower
```

2. Add gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. Use batch normalization:
```python
self.layers = nn.ModuleList([
    nn.Linear(784, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    # ...
])
```

4. Initialize weights with Xavier initialization:
```python
for layer in self.layers:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
```

Try lowering the learning rate first - 0.1 is quite high for SGD!"

**Failure pattern:**
- Agent correctly identified likely cause (gradient explosion from high LR) but didn't DEBUG it
- Suggested fixes without VERIFYING the root cause first
- Did not mention PyTorch's NaN debugging tools: `torch.autograd.set_detect_anomaly()`
- Did not suggest inspecting gradients before NaN appears (to confirm explosion hypothesis)
- Did not explain HOW to verify the fix works (not just "loss is not NaN" but "gradients are stable")
- Threw multiple solutions without systematic diagnosis

**What agent missed:**
- **PyTorch debugging tool:** `torch.autograd.set_detect_anomaly(True)` will pinpoint exact operation producing NaN
  ```python
  with torch.autograd.set_detect_anomaly(True):
      loss.backward()  # Will show stack trace when NaN first appears
  ```
- **Gradient inspection before NaN:**
  ```python
  for name, param in model.named_parameters():
      if param.grad is not None:
          grad_norm = param.grad.norm()
          print(f"{name}: grad norm = {grad_norm:.4f}")
          if grad_norm > 100:  # Warning sign
              print(f"WARNING: Large gradient in {name}")
  ```
- **Systematic NaN debugging:**
  1. Use detect_anomaly() to find WHERE NaN appears
  2. Inspect gradients/activations just before NaN (use hooks)
  3. Verify hypothesis (gradient explosion? activation explosion?)
  4. Apply targeted fix (gradient clipping if gradients explode, reduce LR, etc.)
  5. Verify fix with gradient monitoring
- **Root cause:** LR=0.1 with SGD is too high, causes weight updates too large, leading to numerical instability

---

## Scenario 5: In-Place Operation Breaking Autograd

**Query:** "I'm getting this confusing error during backward pass. The forward pass works fine but backward crashes. Error: 'RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 128]], which is output 0 of ReluBackward0, is at version 2; expected version 1 instead.'"

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_rate = 0.5

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)

        # Manual dropout
        if self.training:
            mask = torch.rand_like(x) > self.dropout_rate
            x *= mask  # In-place multiplication
            x /= (1 - self.dropout_rate)  # In-place division

        x = self.fc2(x)
        return x
```

**Expected Issue:** Agent should recognize in-place operation error pattern:
1. Error message explicitly says "modified by an inplace operation" - read it carefully!
2. Error points to tensor being modified after being used in autograd graph
3. Search code for in-place operations: `x *= ...`, `x /= ...`, `x.add_()`, `x.relu_()`, etc.
4. Identify the culprit: `x *= mask` and `x /= (1 - self.dropout_rate)` are in-place
5. Fix: use out-of-place versions: `x = x * mask` and `x = x / (1 - self.dropout_rate)`
6. Verify: error should disappear after removing in-place ops

**Behavior WITHOUT skill:**

Agent response: "This error happens when PyTorch's autograd graph is broken. Try using `.clone()` to avoid sharing memory:

```python
def forward(self, x):
    x = self.fc1(x)
    x = torch.relu(x)

    if self.training:
        x = x.clone()  # Clone before modification
        mask = torch.rand_like(x) > self.dropout_rate
        x *= mask
        x /= (1 - self.dropout_rate)

    x = self.fc2(x)
    return x
```

Also, consider using PyTorch's built-in dropout instead:
```python
self.dropout = nn.Dropout(0.5)
# Then in forward:
x = self.dropout(x)
```

The built-in dropout handles this correctly!"

**Failure pattern:**
- Agent suggested `.clone()` which treats symptom, not cause (adds memory overhead)
- Did not identify the actual in-place operations (`*=`, `/=`)
- Did not explain WHY in-place operations break autograd (modifies tensor version)
- Suggested using nn.Dropout (good!) but didn't explain the underlying issue
- User won't learn to recognize in-place operation errors in the future

**What agent missed:**
- **Read the error message:** "modified by an inplace operation" tells you exactly what's wrong!
- **In-place operation identification:** Operations with `_` suffix or augmented assignment (`*=`, `+=`, etc.)
- **Why it breaks autograd:** Autograd tracks tensor versions; in-place ops increment version, breaking backward pass
- **Common in-place operations to watch for:**
  - `x += y`, `x *= y`, `x[...] = value` (augmented assignment)
  - `x.relu_()`, `x.add_()`, `x.mul_()` (underscore methods)
  - `x[:] = y` (slice assignment)
- **Proper fix:** Use out-of-place versions: `x = x * mask` instead of `x *= mask`
- **Verification:** Error should disappear after changing to out-of-place ops
- Clone() works but adds memory overhead and doesn't fix the root issue

---

## Scenario 6: Device Mismatch Error

**Query:** "Getting an error about tensors being on different devices. I thought I moved everything to GPU?"

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(100, 256, num_layers=2)
        self.decoder = nn.Linear(256, 10)
        self.register_buffer('positional_encoding', torch.randn(1000, 100))

    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(1)]
        x, _ = self.encoder(x)
        return self.decoder(x[:, -1, :])

device = torch.device('cuda')
model = MyModel().to(device)
data = torch.randn(32, 50, 100).to(device)

output = model(data)  # RuntimeError: Expected all tensors to be on the same device
```

**Expected Issue:** Agent should systematically debug device mismatches:
1. Error says tensors on different devices - need to find WHICH tensors
2. Add device checking at each operation:
   ```python
   print(f"x device: {x.device}")
   print(f"positional_encoding device: {self.positional_encoding.device}")
   ```
3. Discover: positional_encoding is on CPU despite model.to(device)
4. Understand: register_buffer() after model moves to GPU won't auto-move the buffer
5. Fix: register buffer before moving model, OR move buffer explicitly in __init__

**Behavior WITHOUT skill:**

Agent response: "Make sure all tensors are on the same device. Add explicit `.to(device)` calls:

```python
def forward(self, x):
    # Move positional encoding to same device as input
    pos_enc = self.positional_encoding[:x.size(1)].to(x.device)
    x = x + pos_enc
    x, _ = self.encoder(x)
    return self.decoder(x[:, -1, :])
```

This ensures the positional encoding is on the same device as your input!"

**Failure pattern:**
- Agent suggested moving tensor in forward() (works but inefficient - moves every forward pass!)
- Did not DEBUG to find WHY positional_encoding wasn't on GPU
- Did not explain that register_buffer() should auto-move buffers when model.to() is called
- Did not identify the root cause: buffer registered with wrong device
- Solution adds runtime overhead (moving tensor every forward pass)

**What agent missed:**
- **Systematic device debugging:**
  1. Add print statements to check device of each tensor: `print(f"{name}: {tensor.device}")`
  2. Identify which tensor is on wrong device
  3. Trace back WHY it's on wrong device
- **Root cause:** Buffer created on CPU, then not moved by model.to() (likely timing issue in initialization)
- **Proper fix:** Ensure buffer is on correct device during initialization:
  ```python
  def __init__(self):
      super().__init__()
      self.encoder = nn.LSTM(100, 256, num_layers=2)
      self.decoder = nn.Linear(256, 10)
      # Register buffer without data first
      self.register_buffer('positional_encoding', None)

  def reset_parameters(self):
      # Create buffer on correct device
      self.positional_encoding = torch.randn(1000, 100, device=self.encoder.weight.device)
  ```
  Or simpler: don't register as buffer, create as parameter without gradients, or initialize after .to()
- **Verification:** Check all module buffers and parameters are on correct device: `for name, buffer in model.named_buffers(): print(name, buffer.device)`

---

## Scenario 7: Memory Leak in Training Loop

**Query:** "My training starts fine but after a few hundred iterations, I get OOM (out of memory) error. Memory usage keeps growing steadily. I'm clearing gradients properly - what's causing the leak?"

```python
model = LargeModel()
optimizer = torch.optim.Adam(model.parameters())
losses = []

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track losses for plotting
        losses.append(loss)

        if len(losses) % 100 == 0:
            print(f"Step {len(losses)}, Loss: {loss.item():.4f}")
```

**Expected Issue:** Agent should recognize memory accumulation patterns:
1. Memory growing over time suggests accumulation, not just peak usage
2. Check what's being accumulated: `losses` list
3. Identify issue: `losses.append(loss)` keeps tensor with full computation graph
4. Each loss tensor retains references to model outputs and gradients
5. Fix: detach from graph: `losses.append(loss.item())` or `loss.detach().cpu()`
6. Verify: memory should be stable after fix

**Behavior WITHOUT skill:**

Agent response: "Memory leaks in PyTorch training usually come from a few sources:

1. Make sure you're calling `optimizer.zero_grad()` ✓ (you are)

2. Delete intermediate variables:
```python
output = model(batch)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Delete to free memory
del output
del loss
torch.cuda.empty_cache()
```

3. Use gradient accumulation if model is too large:
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Try adding explicit deletes and empty_cache() calls!"

**Failure pattern:**
- Agent did not INVESTIGATE what's accumulating (just suggested generic fixes)
- Suggested `del` and `empty_cache()` which don't address root cause
- Did not identify the actual culprit: `losses.append(loss)` storing tensors with computation graph
- Suggested gradient accumulation which is for different problem (large batch sizes)
- No systematic diagnosis of WHERE memory is leaking

**What agent missed:**
- **Memory leak investigation methodology:**
  1. Identify what's accumulating: check list sizes, check retained tensors
  2. Use memory profiler to see what's growing
  3. Check for unintended tensor retention
- **Common memory leak patterns:**
  - Storing loss/output tensors without `.item()` or `.detach()`
  - Accumulating tensors in lists (for logging, visualization, etc.)
  - Creating new tensors each iteration without freeing old ones
  - Circular references preventing garbage collection
- **Root cause:** `losses.append(loss)` stores tensor with full computation graph
  - Each `loss` tensor keeps reference to model output
  - Model output keeps reference to all intermediate activations
  - Entire forward pass kept in memory for each logged loss!
- **Proper fix:** `losses.append(loss.item())` (stores float, not tensor)
- **Verification:** Monitor memory over iterations: `torch.cuda.memory_allocated()` should be stable

---

## Scenario 8: DataLoader Multiprocessing Deadlock

**Query:** "My training hangs after the first epoch completes. The progress bar for epoch 2 just stops at 0% and never moves. No error message, just hangs. I have to kill the process. What's happening?"

```python
class MyDataset(Dataset):
    def __init__(self):
        self.data = load_large_data()
        self.model_helper = nn.Linear(10, 10)  # Using PyTorch module in dataset

    def __getitem__(self, idx):
        x = self.data[idx]
        # Some preprocessing using the helper
        x = self.model_helper(torch.tensor(x))
        return x.numpy(), self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = MyDataset()
train_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

for epoch in range(10):
    for batch in train_loader:
        # Training code...
        pass
    print(f"Epoch {epoch} complete")
```

**Expected Issue:** Agent should recognize multiprocessing deadlock patterns:
1. Hangs after first epoch + num_workers > 0 suggests multiprocessing issue
2. Dataset contains unpicklable object (nn.Module) or CUDA tensors
3. DataLoader workers fork, try to pickle dataset, deadlock on CUDA context
4. Test hypothesis: set num_workers=0, see if hanging disappears
5. Fix: remove nn.Module from dataset, do preprocessing in __getitem__ without CUDA, or use persistent_workers

**Behavior WITHOUT skill:**

Agent response: "DataLoader hanging issues are frustrating! Try these fixes:

1. Reduce num_workers or set to 0:
```python
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

2. Add timeout:
```python
train_loader = DataLoader(dataset, batch_size=32, num_workers=4, timeout=120)
```

3. Use persistent_workers:
```python
train_loader = DataLoader(dataset, batch_size=32, num_workers=4,
                         persistent_workers=True)
```

4. Check if you're using CUDA tensors in your dataset - move to CPU:
```python
def __getitem__(self, idx):
    x = self.data[idx]
    return torch.tensor(x).cpu(), self.labels[idx]
```

Start with num_workers=0 to see if that fixes it!"

**Failure pattern:**
- Agent suggested reducing workers (treats symptom, not cause) without diagnosis
- Mentioned CUDA tensors (good!) but didn't identify the nn.Module issue
- Did not explain WHY multiprocessing deadlocks happen (pickling, CUDA context, etc.)
- Suggested timeout which won't fix deadlock, just make it error instead of hang
- No systematic approach to diagnosing WHAT in dataset is causing the issue

**What agent missed:**
- **Multiprocessing deadlock diagnosis:**
  1. Test with num_workers=0 to confirm it's multiprocessing issue
  2. Identify unpicklable objects in dataset: nn.Module, CUDA tensors, lambdas, local functions
  3. Understand WHY: DataLoader pickles dataset to send to worker processes
- **Common DataLoader deadlock causes:**
  - nn.Module in dataset (not picklable or causes CUDA context issues)
  - CUDA tensors in dataset (CUDA context can't be forked)
  - Shared resources (file handles, database connections)
  - Large objects being pickled (slow, causes timeout)
- **Root cause:** `self.model_helper = nn.Linear(10, 10)` in dataset
  - Dataset gets pickled to send to worker processes
  - nn.Module with CUDA tensors or tied to main process context causes issues
- **Proper fix:** Remove nn.Module from dataset, do transformations without PyTorch modules, or pre-compute transformations
  ```python
  def __getitem__(self, idx):
      x = self.data[idx]
      # Use numpy/scipy, not PyTorch modules
      x = some_numpy_preprocessing(x)
      return x, self.labels[idx]
  ```
- **Verification:** Should work with num_workers > 0 after removing module

---

## Identified Patterns

### Pattern 1: No Systematic Debugging Methodology
Agents jump to fixes without diagnosing root cause:
- See error → immediate fix attempt without understanding WHY error occurs
- No hypothesis formation: "I think X because Y, let me test with Z"
- Random trial-and-error: "try this, if that doesn't work try that"
- No verification that fix actually addresses root cause (might just hide symptom)

### Pattern 2: Not Reading Error Messages Carefully
Agents skim error messages instead of extracting diagnostic information:
- Miss critical details in error message (e.g., "4x57600" tells you actual tensor shape)
- Don't read full stack trace (miss context about WHERE error originates)
- Ignore specific error patterns (e.g., "device-side assert" = index out of bounds)
- Don't recognize error message patterns (e.g., "version 2; expected version 1" = in-place operation)

### Pattern 3: Unawareness of PyTorch-Specific Debugging Tools
Agents use generic debugging (print statements) instead of PyTorch tools:
- Don't mention `torch.autograd.set_detect_anomaly()` for NaN debugging
- Don't use hooks for intermediate inspection (register_forward_hook, register_hook)
- Don't use assertions to verify understanding (`assert x.shape == expected_shape`)
- Don't check gradient existence/magnitudes when debugging training issues
- Unaware of memory profiling functions (torch.cuda.memory_allocated, memory_summary)

### Pattern 4: Treating Symptoms Instead of Root Causes
Agents fix symptoms without understanding underlying issue:
- Loss is NaN → reduce learning rate (without verifying gradients are exploding)
- Device error → add .to(device) everywhere (without finding WHY tensor is on wrong device)
- Memory leak → add del and empty_cache() (without finding WHAT is accumulating)
- Shape error → add arbitrary reshape (without calculating actual shapes)

### Pattern 5: No Hypothesis Testing
Agents don't form and test hypotheses systematically:
- No "I think X is wrong because Y" reasoning
- No predictions about what investigation will reveal
- No verification of hypothesis before implementing fix
- No testing that fix actually solves the problem (not just makes error disappear)

### Pattern 6: Missing Error Pattern Recognition
Agents don't recognize common PyTorch error patterns:
- "mat1 and mat2 shapes cannot be multiplied" → shape mismatch in linear layer
- "device-side assert" → index out of bounds (usually in embedding)
- "modified by an inplace operation" → in-place op breaking autograd
- "Expected all tensors to be on the same device" → device placement issue
- Constant loss at log(num_classes) → model not learning (random predictions)

### Pattern 7: No Minimal Reproduction
Agents don't isolate the problem:
- Don't narrow down to smallest code that reproduces error
- Don't test components individually (model, data, training loop separately)
- Don't bisect code to find where issue starts
- Don't create minimal test case to verify understanding

---

## What Skill Must Address

### 1. Systematic Debugging Methodology (CRITICAL)

**Phase 1: Reproduce Reliably**
- Fix random seeds for determinism (if error is intermittent)
- Minimize code to smallest reproduction case
- Isolate component causing issue (model? data? training loop?)

**Phase 2: Gather Information**
- Read FULL error message (every word, especially tensor shapes and values)
- Read stack trace to understand WHERE error originates
- Add strategic assertions to verify understanding
- Use PyTorch debugging tools (hooks, detect_anomaly, device checks)

**Phase 3: Form Hypothesis**
- Based on error type and symptoms, what could cause this?
- Predict what investigation will reveal
- Make hypothesis testable (specific prediction)

**Phase 4: Test Hypothesis**
- Add targeted debugging code (not random print statements)
- Verify or reject hypothesis with evidence
- Iterate: if rejected, form new hypothesis

**Phase 5: Fix and Verify**
- Implement minimal fix that addresses root cause (not symptom)
- Verify error is gone AND model works correctly (not just "no error")
- Explain WHY fix works (understanding, not cargo cult)

### 2. PyTorch-Specific Debugging Tools

**For Gradient/NaN Issues:**
- `torch.autograd.set_detect_anomaly(True)` - pinpoints operation producing NaN/Inf
  ```python
  with torch.autograd.set_detect_anomaly(True):
      loss.backward()  # Will raise error with stack trace at NaN source
  ```
- Gradient inspection:
  ```python
  for name, param in model.named_parameters():
      if param.grad is not None:
          print(f"{name}: grad norm = {param.grad.norm():.4f}")
  ```
- `torch.autograd.gradcheck()` - numerical gradient verification

**For Intermediate Inspection:**
- Forward hooks: `module.register_forward_hook(hook_fn)` - inspect outputs
  ```python
  def debug_hook(module, input, output):
      print(f"{module.__class__.__name__}: output shape {output.shape}")
      assert not torch.isnan(output).any(), "NaN detected!"

  for module in model.modules():
      module.register_forward_hook(debug_hook)
  ```
- Backward hooks: `tensor.register_hook(hook_fn)` - inspect gradients
  ```python
  def grad_hook(grad):
      print(f"Gradient: {grad.shape}, mean: {grad.mean():.4f}")
      return grad

  x.register_hook(grad_hook)
  ```

**For Device/Memory Issues:**
- Device checking: `print(f"tensor device: {x.device}")`
- Memory profiling: `torch.cuda.memory_allocated()`, `torch.cuda.memory_summary()`
- Shape assertions: `assert x.shape == expected_shape, f"Got {x.shape}"`

### 3. Common PyTorch Error Patterns (Recognition Framework)

**Shape Errors:**
- "mat1 and mat2 shapes cannot be multiplied (AxB and CxD)"
  → Linear layer input mismatch, check B == C
- "The size of tensor a (X) must match the size of tensor b (Y)"
  → Broadcasting incompatibility, check shapes align

**Device Errors:**
- "Expected all tensors to be on the same device"
  → Some tensor on CPU, others on GPU - add device checks
- "CUDA out of memory"
  → Memory profiling needed, check batch size, model size, gradient accumulation

**Autograd Errors:**
- "one of the variables needed for gradient computation has been modified by an inplace operation"
  → Find `*=`, `+=`, `.add_()`, etc. - replace with out-of-place versions
- "Trying to backward through the graph a second time"
  → Need to use `retain_graph=True` or detach intermediate results

**Numerical Errors:**
- "RuntimeError: Function 'XXX' returned nan values in its 0th output"
  → Use detect_anomaly() to find source
- Constant loss ~log(num_classes)
  → Model outputting random predictions, check optimizer.step() is called

**Data Errors:**
- "CUDA error: device-side assert triggered"
  → Index out of bounds, check data ranges (embedding indices < vocab_size)

**Index/Type Errors:**
- "Expected tensor for argument #1 'indices' to have scalar type Long"
  → Type mismatch, indices must be int64 (target.long())

### 4. Error Classification Decision Tree

**By Symptom:**
1. **Error message** → Read carefully, classify by pattern above
2. **Silent failure (no error but wrong results)** → Check model outputs, gradients, weight updates
3. **Intermittent error** → Data-dependent issue, inspect failing batches
4. **Error after N epochs** → Accumulation issue (memory leak, gradient explosion)

**By Phase:**
1. **Data loading** → Check DataLoader, dataset __getitem__, shapes
2. **Forward pass** → Check shapes, devices, dtypes, intermediate outputs
3. **Backward pass** → Check gradients exist, no in-place ops, detect_anomaly
4. **Optimizer step** → Check optimizer.step() called, learning rate, weight updates

### 5. Common Debugging Pitfalls (Must Explicitly Counter)

❌ **Pitfall 1: Random Trial-and-Error**
- Making changes without hypothesis
- "Let me try changing X" without reasoning about why X would help
- Counter: ALWAYS form hypothesis before making changes

❌ **Pitfall 2: Not Reading Full Error Message**
- Skimming error, missing critical details
- Ignoring stack trace
- Counter: Read EVERY word of error message, especially tensor shapes and operation names

❌ **Pitfall 3: Print-Driven Debugging**
- Adding print statements everywhere without strategy
- Printing in loops (performance impact, overwhelming output)
- Counter: Use assertions and hooks for strategic inspection

❌ **Pitfall 4: Fixing Symptoms, Not Root Causes**
- Making error disappear without understanding why it occurred
- Adding try/except to hide errors
- Counter: Verify understanding of root cause before implementing fix

❌ **Pitfall 5: Not Verifying Fix**
- Assuming fix works because error disappeared
- Not checking that model actually works correctly after fix
- Counter: Verify fix with targeted tests (loss decreases, outputs correct, etc.)

❌ **Pitfall 6: Debugging in Wrong Mode**
- Debugging in train mode when should be eval (or vice versa)
- Debugging without autocast when using mixed precision
- Counter: Match debugging mode to production mode

❌ **Pitfall 7: Not Minimizing Reproduction**
- Trying to debug in full complex codebase
- Not isolating problematic component
- Counter: Create minimal script that reproduces error

❌ **Pitfall 8: Forgetting to Remove Debug Code**
- Leaving detect_anomaly() in production (major performance hit)
- Leaving hooks registered (memory overhead)
- Counter: Remove debug code after fixing, or use environment variable to control

### 6. Hooks Usage Patterns

**When to Use Hooks:**
- Need to inspect intermediate outputs without modifying model code
- Debugging which layer produces NaN
- Checking gradient flow through network
- Monitoring activation/gradient statistics during training

**Forward Hook Pattern:**
```python
def debug_forward_hook(module, input, output):
    print(f"{module.__class__.__name__}:")
    print(f"  Input shape: {input[0].shape if isinstance(input, tuple) else input.shape}")
    print(f"  Output shape: {output.shape}")

    # Check for NaN/Inf
    if torch.isnan(output).any():
        raise RuntimeError(f"NaN detected in {module.__class__.__name__}")
    if torch.isinf(output).any():
        raise RuntimeError(f"Inf detected in {module.__class__.__name__}")

    # Check value range
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

# Register on all modules
handles = []
for name, module in model.named_modules():
    handle = module.register_forward_hook(debug_forward_hook)
    handles.append(handle)

# Run forward pass
output = model(input)

# Remove hooks
for handle in handles:
    handle.remove()
```

**Backward Hook Pattern:**
```python
def debug_grad_hook(grad):
    print(f"Gradient stats:")
    print(f"  Shape: {grad.shape}")
    print(f"  Mean: {grad.mean():.6f}, Std: {grad.std():.6f}")
    print(f"  Min: {grad.min():.6f}, Max: {grad.max():.6f}")
    print(f"  Norm: {grad.norm():.6f}")

    if torch.isnan(grad).any():
        raise RuntimeError("NaN gradient detected!")

    return grad  # Must return gradient

# Register on specific tensor
x = torch.randn(10, requires_grad=True)
x.register_hook(debug_grad_hook)
```

### 7. Rationalization Table (Pressure Resistance)

Must counter these rationalizations:

| Rationalization | Counter-Argument | Red Flag |
|----------------|------------------|----------|
| "Error message is clear, I know what's wrong" | Error message shows symptom; read full trace to find root cause | Jumping to fix without reading full error |
| "User needs quick fix, no time for debugging" | Systematic debugging is FASTER than random trial-and-error | Making changes without hypothesis |
| "This is obviously a shape error, just need to reshape" | Arbitrary reshaping corrupts data; calculate actual shapes needed | Adding reshape without understanding |
| "Let me try changing the learning rate" | Changing parameters without diagnosing root cause wastes time | Suggesting parameter changes without evidence |
| "I'll add prints to see what's happening" | Strategic assertions and hooks are more effective than prints | Adding print statements everywhere |
| "Hooks are too complex for this issue" | Hooks provide targeted inspection without modifying code | Avoiding proper debugging tools |
| "detect_anomaly is slow, skip it" | Only used during debugging, not production - performance doesn't matter | Skipping tools because of performance |
| "Error only happens sometimes, hard to debug" | Fix random seed, isolate failing batch, make deterministic | Giving up on intermittent errors |
| "Just move everything to CPU to avoid CUDA errors" | Moving to CPU hides the root cause; fix device placement instead | Avoiding diagnosis by changing environment |
| "Add try/except to handle the error" | Hiding errors doesn't fix them; find root cause | Using try/except to hide problems |

### 8. Red Flags Checklist

Agent must STOP and debug systematically when:
- ⚠️ Making code changes without hypothesis about root cause
- ⚠️ Suggesting fixes without reading full error message
- ⚠️ Not checking tensor shapes/devices/dtypes when shape/device error occurs
- ⚠️ Suggesting parameter changes (LR, batch size) without profiling/diagnosis
- ⚠️ Adding print statements without clear diagnostic goal
- ⚠️ Implementing fix without predicting what it will reveal
- ⚠️ Not using detect_anomaly() when NaN appears
- ⚠️ Not checking gradients when model not learning
- ⚠️ Treating symptom (add .to(device) everywhere) instead of root cause (why is tensor on wrong device?)
- ⚠️ Not verifying fix actually solves problem (not just makes error disappear)
- ⚠️ Changing multiple things at once (can't isolate what worked)
- ⚠️ Not creating minimal reproduction for complex errors

---

## Conclusion

Agents have basic awareness of debugging (know to check shapes, use print statements) but lack:

1. **Systematic methodology:** No structured approach (reproduce → gather info → hypothesis → test → fix → verify)
2. **Error pattern recognition:** Don't recognize common PyTorch error patterns (in-place ops, device mismatch, etc.)
3. **PyTorch debugging tools:** Unaware of or don't use detect_anomaly(), hooks, assertions, gradient inspection
4. **Root cause analysis:** Fix symptoms instead of root causes (add .to() everywhere instead of finding WHY tensor is on wrong device)
5. **Hypothesis testing:** Make random changes instead of forming testable hypotheses
6. **Verification discipline:** Don't verify fix actually solves problem (just check error disappeared)

The skill must enforce systematic debugging methodology, teach PyTorch-specific debugging tools, provide error pattern recognition framework, and build pressure resistance against random trial-and-error debugging. Agents must learn to DEBUG systematically before FIXING, every time, no exceptions.
