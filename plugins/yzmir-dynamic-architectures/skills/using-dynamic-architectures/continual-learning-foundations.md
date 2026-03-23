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

Formalized by Abraham & Robins (2005):

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
BWT = (1/T-1) * sum(R[T,j] - R[j,j] for j in range(T-1))
# R[i,j] = accuracy on task j after training through task i
# Negative BWT = forgetting
```

**Forward Transfer (FWT)**: How much prior learning helps new tasks

```python
FWT = (1/T-1) * sum(R[i-1,i] - b[i] for i in range(1, T))
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
    def __init__(self, model, c=0.1, epsilon=1e-3):
        self.c = c
        self.epsilon = epsilon
        self.omega = {}  # Importance per parameter
        self.old_params = {}
        self.path_integral = {}  # Running importance

        for n, p in model.named_parameters():
            self.omega[n] = torch.zeros_like(p)
            self.old_params[n] = p.clone().detach()
            self.path_integral[n] = torch.zeros_like(p)

    def update_during_training(self, model):
        """Call after each optimizer step"""
        for n, p in model.named_parameters():
            if p.grad is not None:
                # Accumulate: how much did this param contribute to loss reduction?
                delta = p.detach() - self.old_params[n]
                self.path_integral[n] += -p.grad.detach() * delta
                self.old_params[n] = p.clone().detach()

    def update_omega_at_task_end(self, model):
        """Call when task finishes"""
        for n, p in model.named_parameters():
            delta = (p.detach() - self.old_params[n])**2 + self.epsilon
            self.omega[n] += self.path_integral[n] / delta
            self.path_integral[n].zero_()
            self.old_params[n] = p.clone().detach()

    def penalty(self, model):
        """SI regularization term"""
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.omega[n] * (p - self.old_params[n])**2).sum()
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

```python
class PackNet:
    def __init__(self, model, prune_ratio=0.75):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}  # Binary masks per task

    def train_task(self, task_id, train_data):
        # 1. Train on task (only using free parameters)
        self._apply_mask(task_id - 1)  # Mask previous tasks' params
        train(self.model, train_data)

        # 2. Prune: keep top (1-prune_ratio) of parameters
        mask = self._compute_mask(self.prune_ratio)
        self.masks[task_id] = mask

        # 3. Retrain with pruned network
        self._apply_mask(task_id)
        train(self.model, train_data)

    def _compute_mask(self, prune_ratio):
        """Keep largest magnitude parameters"""
        masks = {}
        for n, p in self.model.named_parameters():
            threshold = torch.quantile(p.abs(), prune_ratio)
            masks[n] = (p.abs() > threshold).float()
        return masks

    def _apply_mask(self, task_id):
        """Zero out parameters used by previous tasks"""
        combined = {}
        for t in range(task_id + 1):
            if t in self.masks:
                for n, m in self.masks[t].items():
                    if n not in combined:
                        combined[n] = torch.zeros_like(m)
                    combined[n] = torch.maximum(combined[n], m)

        for n, p in self.model.named_parameters():
            if n in combined:
                p.data *= (1 - combined[n])  # Zero masked params
```

**Trade-offs:**
- (+) Fixed parameter count (no growth)
- (+) Zero forgetting (parameters are masked, not changed)
- (-) Capacity limit (eventually runs out of free parameters)
- (-) Pruning ratio is a hyperparameter

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
    """Select k samples that maximize coverage"""
    # Compute embeddings
    embeddings = [model.encode(x) for x, y in data]

    # Greedy farthest-point sampling
    selected = [random.randint(0, len(data) - 1)]
    for _ in range(k - 1):
        distances = [
            min(dist(embeddings[i], embeddings[j]) for j in selected)
            for i in range(len(data)) if i not in selected
        ]
        selected.append(distances.index(max(distances)))

    return [data[i] for i in selected]
```

---

## Relevance to Morphogenetic/Dynamic Architectures

### Seeds as Task-Specific Columns

Morphogenetic systems like Esper use "seeds" - new modules that:
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
