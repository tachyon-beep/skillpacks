# Continual Learning Foundations

## Overview

Continual learning addresses the fundamental tension: neural networks trained on new data forget old capabilities. This is **catastrophic forgetting** - not gradual decay, but rapid overwriting. Understanding why this happens and what approaches exist is essential for any dynamic architecture system.

**Core insight:** The problem isn't that networks can't learn sequentially - it's that SGD optimizes for the current objective without considering what parameters meant for previous tasks.

---

## The Catastrophic Forgetting Problem

### Why Naive Fine-Tuning Fails

When you fine-tune a trained model on new data, SGD moves parameters toward the new loss minimum. If this minimum is far from the old one in parameter space, old capabilities are destroyed.

```python
# The failure mode
model = load_pretrained("task_A_expert")  # Good at Task A
train(model, task_B_data)                  # Now good at Task B
evaluate(model, task_A_data)               # Performance collapsed

# Why: Parameters that encoded Task A features were overwritten
```

### The Stability-Plasticity Dilemma

Named in Grossberg's Adaptive Resonance Theory work (1980); Abraham & Robins (2005) is a widely cited later review, not the origin:

- **Stability**: Preserve existing knowledge (resist weight changes)
- **Plasticity**: Learn new information (allow weight changes)

You cannot maximize both. Every continual learning method is a different trade-off.

### Loss Landscape Geometry

Forgetting happens because:

1. **Different optima locations**: Task A and Task B minima are in different regions
2. **Sharp vs flat minima**: Sharp minima overfit to current task; flat minima generalize better across tasks
3. **Parameter drift**: Sequential training walks parameters away from regions that worked for old tasks

```
Task A optimum         Task B optimum
     *                      *
    / \                    / \
   /   \       →→→        /   \
  /     \                /     \
Loss landscape after training on A, then moving toward B
```

### Measuring Forgetting

**Backward Transfer (BWT)**: Performance change on old tasks after learning new ones

```python
BWT = (1 / (T - 1)) * sum(R[T,j] - R[j,j] for j in range(T-1))
# R[i,j] = accuracy on task j after training through task i
# Negative BWT = forgetting
```

**Forward Transfer (FWT)**: How much prior learning helps new tasks

```python
FWT = (1 / (T - 1)) * sum(R[i-1,i] - b[i] for i in range(1, T))
# b[i] = baseline accuracy on task i without prior training
# Positive FWT = beneficial transfer
```

**Average Accuracy**: Performance across all tasks after all training

```python
ACC = (1/T) * sum(R[T,j] for j in range(T))
```

---

## Regularization Approaches

These methods add penalty terms that discourage changing "important" parameters.

### Elastic Weight Consolidation (EWC)

**Paper:** Kirkpatrick et al., 2017 - "Overcoming catastrophic forgetting in neural networks"

**Core idea:** Some parameters matter more for old tasks. Penalize changing those.

**Mechanism:** Use Fisher Information to estimate parameter importance.

```python
# After training on task A, compute Fisher Information
fisher_A = compute_fisher(model, task_A_data)

# When training on task B, add EWC penalty
def ewc_loss(model, task_B_loss, lambda_ewc=1000):
    ewc_penalty = 0
    for name, param in model.named_parameters():
        # Penalize deviation from Task A parameters
        ewc_penalty += (fisher_A[name] * (param - theta_A[name])**2).sum()
    return task_B_loss + (lambda_ewc / 2) * ewc_penalty

def compute_fisher(model, data):
    """Diagonal Fisher Information Matrix"""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x, y in data:
        model.zero_grad()
        output = model(x)
        # Sample from output distribution (or use labels)
        log_prob = F.log_softmax(output, dim=1)
        sampled = log_prob.gather(1, y.unsqueeze(1))
        sampled.sum().backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2
    # Normalize
    for n in fisher:
        fisher[n] /= len(data)
    return fisher
```

**Trade-offs:**
- (+) Simple to implement
- (+) No extra data storage (just Fisher + old params per task)
- (-) Fisher is approximate (diagonal only)
- (-) Scales poorly with many tasks (accumulating constraints)
- (-) Requires knowing task boundaries

### Synaptic Intelligence (SI)

**Paper:** Zenke et al., 2017 - "Continual Learning Through Synaptic Intelligence"

**Core idea:** Track importance online during training, not just at task end.

**Mechanism:** Accumulate gradient contributions to loss reduction.

```python
class SynapticIntelligence:
    """
    SI needs TWO parameter snapshots, and conflating them is the classic bug:

      prev_step_params  — refreshed EVERY optimizer step; used only to get the
                          per-step displacement for the path integral.
      task_start_params — frozen for the whole task; it is both the denominator
                          reference (TOTAL displacement over the task) and the
                          anchor θ* the quadratic penalty pulls back toward.

    Use one dict for both and you get omega ≈ path_integral / ε (the denominator
    becomes a single step's displacement) and a penalty that is ~0 always
    (because the anchor is where you were one step ago).
    """
    def __init__(self, model, c=0.1, epsilon=1e-3):
        self.c = c
        self.epsilon = epsilon
        self.omega = {}              # Accumulated importance per parameter
        self.prev_step_params = {}   # θ at the previous optimizer step
        self.task_start_params = {}  # θ* — start of current task / end of previous
        self.path_integral = {}      # Running -g · Δθ for this task

        for n, p in model.named_parameters():
            self.omega[n] = torch.zeros_like(p)
            self.prev_step_params[n] = p.clone().detach()
            self.task_start_params[n] = p.clone().detach()
            self.path_integral[n] = torch.zeros_like(p)

    def update_during_training(self, model):
        """Call after each optimizer step"""
        for n, p in model.named_parameters():
            if p.grad is not None:
                # Per-step contribution to loss reduction
                delta = p.detach() - self.prev_step_params[n]
                self.path_integral[n] += -p.grad.detach() * delta
            self.prev_step_params[n] = p.clone().detach()   # step-local ONLY

    def update_omega_at_task_end(self, model):
        """Call when task finishes"""
        for n, p in model.named_parameters():
            # Denominator: TOTAL displacement over the task, not one step's
            total_delta = (p.detach() - self.task_start_params[n])**2 + self.epsilon
            self.omega[n] += (self.path_integral[n] / total_delta).clamp(min=0)
            self.path_integral[n].zero_()
            # New anchor for the next task's penalty
            self.task_start_params[n] = p.clone().detach()
            self.prev_step_params[n] = p.clone().detach()

    def penalty(self, model):
        """SI regularization term — anchored at the END of the previous task"""
        loss = 0
        for n, p in model.named_parameters():
            loss = loss + (self.omega[n] * (p - self.task_start_params[n])**2).sum()
        return self.c * loss
```

**Trade-offs:**
- (+) Online importance estimation (no separate Fisher computation)
- (+) More accurate than EWC in some settings
- (-) More complex bookkeeping
- (-) Still needs task boundaries for omega update

### Memory Aware Synapses (MAS)

**Paper:** Aljundi et al., 2018

**Core idea:** Use gradient magnitude as importance proxy, computed on unlabeled data.

```python
def compute_mas_importance(model, data):
    """MAS doesn't need labels - uses output magnitude"""
    importance = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x in data:  # No labels needed
        model.zero_grad()
        output = model(x)
        # Use L2 norm of output as "importance" signal
        output.norm(2).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.data.abs()
    for n in importance:
        importance[n] /= len(data)
    return importance
```

**Trade-offs:**
- (+) No labels needed (can use unlabeled data)
- (+) Task-agnostic importance measure
- (-) May not capture task-specific importance as well as EWC

### Comparison Table

| Method | Importance Measure | When Computed | Labels Needed | Task Boundaries |
|--------|-------------------|---------------|---------------|-----------------|
| EWC | Fisher Information | End of task | Yes | Yes |
| SI | Path integral | During training | Yes | Yes |
| MAS | Gradient magnitude | Any time | No | No |

---

## Architectural Approaches

Instead of constraining parameters, allocate new capacity for new tasks.

### Progressive Neural Networks

**Paper:** Rusu et al., 2016 - "Progressive Neural Networks"

**Core idea:** Freeze old columns, add new column with lateral connections.

```
Task 1:  [Column 1] (frozen after training)
              ↓ lateral connections
Task 2:  [Column 1] → [Column 2] (frozen after training)
              ↓           ↓
Task 3:  [Column 1] → [Column 2] → [Column 3]
```

```python
class ProgressiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.columns = nn.ModuleList()
        self.lateral = nn.ModuleList()

    def add_column(self, column_config):
        """Add new column for new task"""
        new_col = self._build_column(column_config)

        # Lateral connections from all previous columns
        if len(self.columns) > 0:
            laterals = nn.ModuleList([
                nn.Linear(prev_col.hidden_size, new_col.hidden_size)
                for prev_col in self.columns
            ])
            self.lateral.append(laterals)

        self.columns.append(new_col)

        # Freeze all previous columns
        for i, col in enumerate(self.columns[:-1]):
            for param in col.parameters():
                param.requires_grad = False

    def forward(self, x, task_id):
        # Only use columns up to task_id
        outputs = []
        for i, col in enumerate(self.columns[:task_id + 1]):
            h = col.layer1(x)
            if i > 0:
                # Add lateral inputs from previous columns
                for j, lateral in enumerate(self.lateral[i-1]):
                    h = h + lateral(outputs[j])
            h = F.relu(h)
            outputs.append(h)
        return self.columns[task_id].head(outputs[-1])
```

**Trade-offs:**
- (+) Zero forgetting (old columns frozen)
- (+) Positive forward transfer via laterals
- (-) Linear parameter growth with tasks
- (-) Inference requires all columns (compute grows)

### PackNet

**Paper:** Mallya & Lazebnik, 2018 - "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"

**Core idea:** Prune network after each task, use freed capacity for next task.

**The invariant that makes PackNet work:** parameters owned by earlier tasks are **frozen, never modified**. Forgetting is zero because those weights are bit-identical after later training — not because they are masked out at the end. The common implementation bug is to *zero* previous tasks' weights ("masking" read as "erase"), which destroys exactly the knowledge the method claims to preserve. Freeze via gradient masking; leave the values alone.

```python
class PackNet:
    """
    Mallya & Lazebnik, 2018.

    Invariant: once task t's parameters are elected, they are frozen forever.
    Verify it — after training task t+1, assert task t's owned parameters are
    unchanged. If they moved, the implementation is wrong, not the method.
    """
    def __init__(self, model, prune_ratio=0.75):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}   # task_id -> {param_name: 1.0 where OWNED by that task}

    def _owned_before(self, task_id):
        """Union of masks for all tasks < task_id (1 = owned and frozen)."""
        owned = {}
        for t, mask in self.masks.items():
            if t >= task_id:
                continue
            for n, m in mask.items():
                owned[n] = m.clone() if n not in owned else torch.maximum(owned[n], m)
        for n, p in self.model.named_parameters():
            owned.setdefault(n, torch.zeros_like(p))
        return owned

    def _restrict_updates_to(self, trainable):
        """
        Gradient hooks: only parameters with a 1 in `trainable` receive updates.
        This is the freeze — weight VALUES are never touched.
        Note: with momentum/weight-decay optimizers, also confirm the optimizer
        cannot move a zero-gradient parameter (decoupled weight decay can).
        """
        return [p.register_hook(lambda g, keep=trainable[n]: g * keep)
                for n, p in self.model.named_parameters()]

    def _compute_mask(self, owned):
        """Elect this task's parameters: largest-magnitude among the FREE ones."""
        masks = {}
        for n, p in self.model.named_parameters():
            free = 1.0 - owned[n]
            k = int((1.0 - self.prune_ratio) * free.sum().item())   # how many to keep
            if k < 1:
                masks[n] = torch.zeros_like(p)      # no capacity left for this tensor
                continue
            scores = (p.detach().abs() * free).flatten()
            threshold = scores.topk(k).values.min()
            masks[n] = ((p.detach().abs() >= threshold) & (free > 0)).float()
        return masks

    def train_task(self, task_id, train_data):
        owned = self._owned_before(task_id)

        # 1. Train on the FREE parameters only (earlier tasks frozen by hook)
        free = {n: 1.0 - owned[n] for n in owned}
        handles = self._restrict_updates_to(free)
        train(self.model, train_data)
        for h in handles:
            h.remove()

        # 2. Prune: elect the top (1 - prune_ratio) of the free parameters
        mask = self._compute_mask(owned)
        self.masks[task_id] = mask

        # 3. Zero only the free parameters this task did NOT elect (they return to
        #    the free pool for future tasks). Owned + elected weights are untouched.
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.mul_(owned[n] + mask[n])

        # 4. Retrain to recover pruning damage — updating THIS task's weights only.
        handles = self._restrict_updates_to(mask)
        train(self.model, train_data)
        for h in handles:
            h.remove()

    def active_mask_for_inference(self, task_id):
        """At test time on task t, use the union of masks for tasks <= t."""
        return self._owned_before(task_id + 1)
```

Applied to a two-task toy problem, this holds the invariant: every parameter elected by task 1 is bit-identical after task 2 trains, and the two tasks' masks are disjoint. In practice mask only weight tensors (biases and norm parameters are usually shared or handled separately), and note that inference requires knowing which task you are on — PackNet is a task-incremental, not class-incremental, method.

**Trade-offs:**
- (+) Fixed parameter count (no growth)
- (+) Zero forgetting — but only because earlier tasks' weights are frozen, not zeroed
- (-) Capacity limit (eventually runs out of free parameters)
- (-) Pruning ratio is a hyperparameter
- (-) Needs the task identity at inference time to pick the right mask

### Dynamically Expandable Networks (DEN)

**Paper:** Yoon et al., 2018

**Core idea:** Selectively retrain, split neurons, or expand when needed.

```python
# Simplified DEN logic
def train_task_den(model, task_data, threshold_expand=0.1):
    # 1. Selective retraining: only retrain neurons relevant to new task
    relevant = identify_relevant_neurons(model, task_data)
    freeze_except(model, relevant)
    train(model, task_data)

    # 2. If performance insufficient, expand network
    if eval(model, task_data) < threshold_expand:
        new_neurons = add_neurons(model, count=estimate_needed())
        train_new_only(model, task_data, new_neurons)

    # 3. Split neurons that became too task-specific
    split_overloaded_neurons(model, task_data)
```

**Trade-offs:**
- (+) Adaptive expansion (grows only when needed)
- (+) Can reuse capacity when appropriate
- (-) Complex decision logic
- (-) Splitting heuristics can be fragile

---

## Rehearsal Approaches

Store or generate old data to mix with new training.

### Experience Replay

Store subset of old data, replay during new training.

```python
class ReplayBuffer:
    def __init__(self, capacity=10000, samples_per_task=1000):
        self.capacity = capacity
        self.samples_per_task = samples_per_task
        self.buffer = []

    def add_task_samples(self, task_id, data):
        """Store representative samples from completed task"""
        # Random selection or use coreset selection
        indices = random.sample(range(len(data)), self.samples_per_task)
        for i in indices:
            self.buffer.append((task_id, data[i]))

        # Evict if over capacity (oldest first or balanced)
        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_replay_batch(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

def train_with_replay(model, new_data, replay_buffer, replay_ratio=0.5):
    for batch in new_data:
        # Mix new data with replay
        replay_batch = replay_buffer.get_replay_batch(
            int(len(batch) * replay_ratio)
        )
        combined = merge_batches(batch, replay_batch)
        train_step(model, combined)
```

**Trade-offs:**
- (+) Simple and effective
- (+) Works with any architecture
- (-) Storage requirements grow with tasks
- (-) Privacy concerns (storing old data)

### Generative Replay

Train a generator to produce old data instead of storing it.

```python
class GenerativeReplay:
    def __init__(self, generator, solver):
        self.generator = generator  # Generates (x, y) for old tasks
        self.solver = solver        # The actual model being trained

    def train_task(self, task_id, new_data):
        if task_id > 0:
            # Generate pseudo-data for old tasks
            old_data = self.generator.sample(n=len(new_data))

            # Train solver on new + generated old
            train_interleaved(self.solver, new_data, old_data)
        else:
            train(self.solver, new_data)

        # Update generator to also produce new task data
        train(self.generator, new_data)
```

**Trade-offs:**
- (+) Constant memory (generator size fixed)
- (+) No privacy concerns (no real data stored)
- (-) Generator must be good enough to capture data distribution
- (-) Generator training adds complexity
- (-) Quality degrades over many tasks (error accumulation)

### Coreset Selection

Instead of random sampling, select maximally informative samples.

```python
def select_coreset(data, model, k):
    """Select k samples that maximize coverage (greedy farthest-point sampling)"""
    # Compute embeddings
    embeddings = [model.encode(x) for x, y in data]

    selected = [random.randint(0, len(data) - 1)]
    remaining = set(range(len(data))) - set(selected)

    for _ in range(min(k, len(data)) - 1):
        # Track the ORIGINAL index alongside the distance. Building a filtered
        # list and then using `distances.index(max(...))` returns a position in
        # the filtered list, which is a *different* sample once anything has
        # been removed — a silent wrong-sample bug that still "works".
        best_idx, best_dist = None, -float("inf")
        for i in remaining:
            d = min(dist(embeddings[i], embeddings[j]) for j in selected)
            if d > best_dist:
                best_idx, best_dist = i, d
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [data[i] for i in selected]
```

---

## Relevance to Morphogenetic/Dynamic Architectures

### Seeds as Task-Specific Columns

Morphogenetic RL systems use "seeds" - new modules that:
1. Train in isolation (like Progressive columns)
2. Connect to existing host (like lateral connections)
3. Get frozen when integrated (like PackNet masking)

**Mapping:**

| Continual Learning | Morphogenetic System |
|--------------------|---------------------|
| Task boundary | Seed lifecycle transition |
| New column/capacity | Germinated seed |
| Lateral connections | Seed input from host stream |
| Column freezing | Seed fossilization |
| Pruning | Seed culling/embargo |

### Gradient Isolation as Architectural Approach

The "gradient isolation" technique in morphogenetic systems is an architectural solution:
- Host parameters frozen relative to seed's training
- Seed learns from host errors (residual learning)
- Integration is gradual (alpha blending)

This is closest to **Progressive Neural Networks** but with:
- Dynamic, not pre-defined, expansion points
- Gradual integration, not binary freeze
- Quality gates before permanent integration

### Choosing an Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Few tasks, compute-cheap | Progressive Neural Networks |
| Many tasks, memory-limited | EWC or SI + PackNet |
| Tasks arrive continuously | SI (online importance) + Replay |
| Privacy-sensitive | Generative Replay or pure regularization |
| Dynamic capacity | Morphogenetic (seeds + lifecycle) |

---

## Implementation Checklist

When implementing continual learning:

- [ ] Define task boundaries (or use online method if boundaries unclear)
- [ ] Choose metric: backward transfer, forward transfer, average accuracy
- [ ] Select approach based on constraints (memory, compute, privacy)
- [ ] Implement importance measurement (Fisher, SI, MAS) or capacity allocation
- [ ] Consider hybrid (regularization + small replay buffer)
- [ ] Measure forgetting explicitly (don't just track new task performance)

---

## References

- Kirkpatrick et al., 2017 - "Overcoming catastrophic forgetting in neural networks" (EWC)
- Zenke et al., 2017 - "Continual Learning Through Synaptic Intelligence" (SI)
- Aljundi et al., 2018 - "Memory Aware Synapses" (MAS)
- Rusu et al., 2016 - "Progressive Neural Networks"
- Mallya & Lazebnik, 2018 - "PackNet"
- Yoon et al., 2018 - "Lifelong Learning with Dynamically Expandable Networks" (DEN)
- Lopez-Paz & Ranzato, 2017 - "Gradient Episodic Memory" (GEM)
- Shin et al., 2017 - "Continual Learning with Deep Generative Replay"
