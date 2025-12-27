# Dynamic Architecture Patterns

## Overview

Dynamic architectures change structure during training - adding capacity when needed, removing it when unhelpful. This differs from Neural Architecture Search (NAS), which searches once before training. Dynamic architectures adapt continuously.

**Core insight:** The right architecture depends on training progress. Early training may need exploration capacity; late training may benefit from pruning. Static architectures are a guess; dynamic architectures are responsive.

---

## Why Dynamic Architecture

### Static Architecture Limitations

When you fix architecture before training:

1. **Capacity guessing**: Too small = underfitting, too large = overfitting + waste
2. **Uniform allocation**: Same capacity everywhere, even where not needed
3. **No adaptation**: Cannot respond to what training reveals about the problem

### Dynamic Architecture Benefits

1. **Demand-driven growth**: Add capacity where training signal indicates need
2. **Efficiency**: Remove capacity that isn't contributing
3. **Exploration**: Try different structures, keep what works

### Distinction from NAS

| Aspect | Neural Architecture Search | Dynamic Architecture |
|--------|---------------------------|---------------------|
| When | Before training | During training |
| How | Search space + optimization | Grow/prune rules or learned |
| Result | Fixed architecture | Evolving architecture |
| Cost | Expensive search phase | Integrated with training |

---

## Growth Patterns

### Slot-Based Expansion

Pre-define attachment points where new modules can be added.

```python
class SlottedNetwork(nn.Module):
    def __init__(self, base_network, slot_configs):
        super().__init__()
        self.base = base_network
        self.slots = nn.ModuleDict()
        self.slot_configs = slot_configs  # Where slots can attach

        # Initialize empty slots
        for slot_id, config in slot_configs.items():
            self.slots[slot_id] = None  # Empty slot

    def add_module_to_slot(self, slot_id, module):
        """Instantiate module in empty slot"""
        if self.slots[slot_id] is not None:
            raise ValueError(f"Slot {slot_id} already occupied")
        self.slots[slot_id] = module

    def remove_module_from_slot(self, slot_id):
        """Remove module, return to empty slot"""
        module = self.slots[slot_id]
        self.slots[slot_id] = None
        return module

    def forward(self, x):
        # Base network with slot injections
        for layer_name, layer in self.base.named_children():
            x = layer(x)

            # Check for active slots at this layer
            for slot_id, config in self.slot_configs.items():
                if config['attach_after'] == layer_name:
                    if self.slots[slot_id] is not None:
                        slot_out = self.slots[slot_id](x)
                        x = x + slot_out  # Or other combination

        return x
```

**Slot addressing patterns:**
- **Position-based**: `r0c0` = row 0, column 0 (grid topology)
- **Layer-based**: `after_layer_3`, `parallel_to_encoder`
- **Functional**: `residual_slot`, `attention_slot`

### Layer Widening (Net2Net Style)

Add neurons/channels to existing layers without losing learned features.

```python
def widen_layer(layer, new_width, noise_std=0.01):
    """
    Widen a linear layer while preserving function.
    Based on Net2Net (Chen et al., 2015)
    """
    old_width = layer.out_features
    if new_width <= old_width:
        return layer

    # Create new wider layer
    new_layer = nn.Linear(layer.in_features, new_width)

    # Copy existing weights
    new_layer.weight.data[:old_width] = layer.weight.data
    new_layer.bias.data[:old_width] = layer.bias.data

    # New neurons: copy random existing + add noise
    for i in range(old_width, new_width):
        source_idx = random.randint(0, old_width - 1)
        new_layer.weight.data[i] = layer.weight.data[source_idx].clone()
        new_layer.bias.data[i] = layer.bias.data[source_idx].clone()
        # Add noise to break symmetry
        new_layer.weight.data[i] += torch.randn_like(new_layer.weight.data[i]) * noise_std
        new_layer.bias.data[i] += torch.randn_like(new_layer.bias.data[i]) * noise_std

    return new_layer

def widen_conv(conv, new_channels, noise_std=0.01):
    """Widen convolutional layer"""
    old_channels = conv.out_channels
    if new_channels <= old_channels:
        return conv

    new_conv = nn.Conv2d(
        conv.in_channels, new_channels,
        conv.kernel_size, conv.stride, conv.padding
    )

    # Copy existing filters
    new_conv.weight.data[:old_channels] = conv.weight.data
    if conv.bias is not None:
        new_conv.bias.data[:old_channels] = conv.bias.data

    # New filters: copy + noise
    for i in range(old_channels, new_channels):
        source_idx = random.randint(0, old_channels - 1)
        new_conv.weight.data[i] = conv.weight.data[source_idx].clone()
        new_conv.weight.data[i] += torch.randn_like(new_conv.weight.data[i]) * noise_std
        if conv.bias is not None:
            new_conv.bias.data[i] = conv.bias.data[source_idx] + noise_std * torch.randn(1)

    return new_conv
```

**Critical:** When widening layer N, must also update layer N+1's input dimension.

### Depth Extension

Insert new layers into the network.

```python
def insert_layer(model, position, new_layer, init='identity'):
    """
    Insert layer at position. Initialize to identity to preserve function.
    """
    if init == 'identity':
        # Initialize to approximate identity
        if isinstance(new_layer, nn.Linear):
            nn.init.eye_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
        elif isinstance(new_layer, nn.Conv2d):
            # Identity for 1x1 conv
            if new_layer.kernel_size == (1, 1):
                nn.init.dirac_(new_layer.weight)
            # For larger kernels, center the identity
            else:
                nn.init.zeros_(new_layer.weight)
                center = new_layer.kernel_size[0] // 2
                for i in range(min(new_layer.in_channels, new_layer.out_channels)):
                    new_layer.weight.data[i, i, center, center] = 1.0
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias)

    elif init == 'zero':
        # Zero init: new layer contributes nothing initially
        for param in new_layer.parameters():
            nn.init.zeros_(param)

    # Insert into model (implementation depends on model structure)
    # For nn.Sequential:
    layers = list(model.children())
    layers.insert(position, new_layer)
    return nn.Sequential(*layers)
```

### Branching (Parallel Paths)

Add parallel computation paths that can specialize.

```python
class BranchableLayer(nn.Module):
    def __init__(self, base_layer, max_branches=4):
        super().__init__()
        self.branches = nn.ModuleList([base_layer])
        self.max_branches = max_branches
        self.branch_weights = nn.Parameter(torch.ones(1))

    def add_branch(self, new_branch, init='zero'):
        if len(self.branches) >= self.max_branches:
            raise ValueError("Max branches reached")

        if init == 'zero':
            for param in new_branch.parameters():
                nn.init.zeros_(param)

        self.branches.append(new_branch)
        # Expand weights
        new_weights = torch.ones(len(self.branches))
        new_weights[:-1] = self.branch_weights.data
        new_weights[-1] = 0.0 if init == 'zero' else 1.0
        self.branch_weights = nn.Parameter(new_weights)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        weights = F.softmax(self.branch_weights, dim=0)
        return sum(w * o for w, o in zip(weights, outputs))
```

---

## Pruning Patterns

### Magnitude Pruning

Remove weights/neurons with smallest magnitude.

```python
def magnitude_prune(model, prune_ratio, granularity='weight'):
    """
    Prune smallest magnitude parameters.
    granularity: 'weight' (unstructured) or 'neuron' (structured)
    """
    if granularity == 'weight':
        # Collect all weights
        all_weights = torch.cat([p.view(-1) for p in model.parameters()])
        threshold = torch.quantile(all_weights.abs(), prune_ratio)

        # Create masks
        masks = {}
        for name, param in model.named_parameters():
            masks[name] = (param.abs() > threshold).float()
            param.data *= masks[name]  # Zero out pruned weights

        return masks

    elif granularity == 'neuron':
        # For each layer, compute neuron importance as L2 norm
        masks = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Importance = L2 norm of outgoing weights
                importance = module.weight.norm(dim=1)
                threshold = torch.quantile(importance, prune_ratio)
                mask = (importance > threshold).float()
                masks[name] = mask

                # Zero out pruned neurons
                module.weight.data *= mask.unsqueeze(1)
                if module.bias is not None:
                    module.bias.data *= mask

        return masks
```

### Gradient-Based Pruning

Remove parameters with low gradient signal (less important for current task).

```python
class GradientImportancePruner:
    def __init__(self, model):
        self.model = model
        self.importance = {n: torch.zeros_like(p)
                          for n, p in model.named_parameters()}

    def accumulate_importance(self, loss):
        """Call after computing loss, before optimizer step"""
        loss.backward(retain_graph=True)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Importance = accumulated gradient magnitude
                self.importance[name] += param.grad.abs()

    def prune(self, prune_ratio):
        """Prune parameters with lowest accumulated importance"""
        all_importance = torch.cat([v.view(-1) for v in self.importance.values()])
        threshold = torch.quantile(all_importance, prune_ratio)

        masks = {}
        for name, param in self.model.named_parameters():
            mask = (self.importance[name] > threshold).float()
            masks[name] = mask
            param.data *= mask

        return masks
```

### Lottery Ticket Pruning

Find sparse subnetwork, reset to initialization, retrain.

```python
class LotteryTicketPruner:
    def __init__(self, model):
        self.model = model
        # Save initial weights
        self.initial_weights = {n: p.clone() for n, p in model.named_parameters()}
        self.masks = None

    def train_and_prune(self, train_fn, prune_ratio, iterations=3):
        """Iterative magnitude pruning"""
        per_iteration_prune = 1 - (1 - prune_ratio) ** (1 / iterations)

        for i in range(iterations):
            # Train
            train_fn(self.model)

            # Prune by magnitude
            self.masks = magnitude_prune(self.model, per_iteration_prune)

            # Reset to initial weights (but keep mask)
            for name, param in self.model.named_parameters():
                param.data = self.initial_weights[name].clone()
                param.data *= self.masks[name]

        # Final training with pruned network
        train_fn(self.model)
```

### Structured vs Unstructured Pruning

| Type | Granularity | Speedup | Accuracy |
|------|-------------|---------|----------|
| Unstructured | Individual weights | Needs sparse hardware | Better |
| Structured | Neurons/channels/heads | Real speedup on GPU | Worse |

```python
# Unstructured: sparse weights, irregular pattern
weight = [[0, 0.5, 0], [0.3, 0, 0.7], [0, 0.2, 0]]

# Structured: remove entire neurons
weight = [[0, 0, 0], [0.3, 0.5, 0.7], [0, 0, 0]]  # Neuron 0 and 2 removed
# Can actually remove rows, reducing matrix size
```

---

## Trigger Conditions

### Loss Plateau Detection

Grow when loss stops improving.

```python
class PlateauDetector:
    def __init__(self, patience=10, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.history = []
        self.best_loss = float('inf')
        self.wait = 0

    def update(self, loss):
        self.history.append(loss)

        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.wait = 0
            return False  # Still improving
        else:
            self.wait += 1
            return self.wait >= self.patience  # Plateau detected

    def should_grow(self, loss):
        """Returns True if growth is warranted"""
        return self.update(loss)
```

### Contribution Metrics

Prune modules that don't contribute.

```python
class ContributionTracker:
    def __init__(self, model, baseline_metric_fn):
        self.model = model
        self.baseline_metric = baseline_metric_fn
        self.contributions = {}

    def measure_contribution(self, module_name, data):
        """
        Counterfactual: what's the metric with vs without this module?
        """
        # Get baseline
        baseline = self.baseline_metric(self.model, data)

        # Disable module
        module = dict(self.model.named_modules())[module_name]
        original_forward = module.forward
        module.forward = lambda x: torch.zeros_like(original_forward(x))

        # Measure without
        without = self.baseline_metric(self.model, data)

        # Restore
        module.forward = original_forward

        # Contribution = baseline - without (positive = helps)
        contribution = baseline - without
        self.contributions[module_name] = contribution
        return contribution

    def get_pruneable(self, threshold=0.0):
        """Modules with contribution below threshold"""
        return [name for name, contrib in self.contributions.items()
                if contrib < threshold]
```

### Budget Constraints

Trigger growth/pruning based on resource limits.

```python
class BudgetManager:
    def __init__(self, param_budget, flops_budget=None):
        self.param_budget = param_budget
        self.flops_budget = flops_budget

    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def can_grow(self, model, new_module):
        current = self.count_params(model)
        additional = sum(p.numel() for p in new_module.parameters())
        return current + additional <= self.param_budget

    def must_prune(self, model):
        return self.count_params(model) > self.param_budget

    def headroom(self, model):
        """How much capacity remains"""
        return self.param_budget - self.count_params(model)
```

---

## Capacity Scheduling

### Start Small, Grow as Needed

```python
class GrowAsNeeded:
    def __init__(self, model, growth_manager, plateau_detector):
        self.model = model
        self.growth = growth_manager
        self.plateau = plateau_detector

    def step(self, loss, epoch):
        if self.plateau.should_grow(loss):
            if self.growth.can_grow(self.model):
                new_module = self.growth.create_growth_module()
                self.growth.add_to_model(self.model, new_module)
                self.plateau.reset()  # Reset patience after growth
                return True
        return False
```

### Overparameterize Then Prune

```python
class OverparameterizeThenPrune:
    def __init__(self, model, target_sparsity=0.9):
        self.model = model
        self.target_sparsity = target_sparsity
        self.phase = 'train'  # 'train' -> 'prune' -> 'finetune'

    def step(self, epoch, total_epochs):
        train_epochs = int(total_epochs * 0.6)
        prune_epochs = int(total_epochs * 0.2)

        if epoch < train_epochs:
            self.phase = 'train'
        elif epoch < train_epochs + prune_epochs:
            self.phase = 'prune'
            # Gradual pruning
            progress = (epoch - train_epochs) / prune_epochs
            current_sparsity = progress * self.target_sparsity
            magnitude_prune(self.model, current_sparsity)
        else:
            self.phase = 'finetune'
            # Final finetuning with fixed mask
```

### Alternating Grow/Prune Phases

```python
class GrowPruneCycle:
    def __init__(self, grow_epochs=10, prune_epochs=5, stabilize_epochs=5):
        self.grow_epochs = grow_epochs
        self.prune_epochs = prune_epochs
        self.stabilize_epochs = stabilize_epochs
        self.cycle_length = grow_epochs + prune_epochs + stabilize_epochs

    def get_phase(self, epoch):
        position = epoch % self.cycle_length
        if position < self.grow_epochs:
            return 'grow'
        elif position < self.grow_epochs + self.prune_epochs:
            return 'prune'
        else:
            return 'stabilize'

    def step(self, epoch, model, growth_manager, pruner):
        phase = self.get_phase(epoch)

        if phase == 'grow':
            # Allow growth actions
            growth_manager.enabled = True
            pruner.enabled = False
        elif phase == 'prune':
            # Allow pruning actions
            growth_manager.enabled = False
            pruner.enabled = True
            pruner.prune_step(model)
        else:
            # Stabilize: no structural changes
            growth_manager.enabled = False
            pruner.enabled = False
```

---

## Slot Semantics

### Canonical Slot Addressing

```python
class SlotTopology:
    """
    Grid-based slot addressing.
    Rows = depth in network
    Columns = parallel slots at same depth
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.slots = {}

        for r in range(rows):
            for c in range(cols):
                slot_id = f"r{r}c{c}"
                self.slots[slot_id] = {
                    'state': 'dormant',
                    'module': None,
                    'depth': r,
                    'parallel_idx': c
                }

    def get_slot(self, slot_id):
        return self.slots.get(slot_id)

    def slots_at_depth(self, depth):
        return [s for s in self.slots.values() if s['depth'] == depth]

    def available_slots(self):
        return [sid for sid, s in self.slots.items() if s['state'] == 'dormant']
```

### Slot States

| State | Meaning | Transitions To |
|-------|---------|----------------|
| Dormant | Empty, available | Germinated |
| Germinated | Module created, not training | Training |
| Training | Module actively learning | Integrated, Pruned |
| Integrated | Module is part of host | (terminal) |
| Pruned | Module removed | Embargoed |
| Embargoed | Slot on cooldown | Dormant |

### Cooldown and Recycling

```python
class SlotLifecycle:
    def __init__(self, embargo_duration=100):
        self.embargo_duration = embargo_duration
        self.embargo_timers = {}

    def prune_slot(self, slot_id, current_step):
        """Move slot to embargo state"""
        self.embargo_timers[slot_id] = current_step + self.embargo_duration
        return 'embargoed'

    def check_embargo(self, slot_id, current_step):
        """Check if embargo is over"""
        if slot_id in self.embargo_timers:
            if current_step >= self.embargo_timers[slot_id]:
                del self.embargo_timers[slot_id]
                return 'dormant'  # Available again
            return 'embargoed'
        return None

    def available_for_growth(self, slots, current_step):
        """Return slots that can accept new modules"""
        available = []
        for slot_id, slot in slots.items():
            if slot['state'] == 'dormant':
                available.append(slot_id)
            elif slot['state'] == 'embargoed':
                if self.check_embargo(slot_id, current_step) == 'dormant':
                    slot['state'] = 'dormant'
                    available.append(slot_id)
        return available
```

---

## Implementation Checklist

When implementing dynamic architecture:

- [ ] Define growth points (slots, layer positions, or branching)
- [ ] Choose growth trigger (plateau, budget, learned)
- [ ] Choose pruning criterion (magnitude, gradient, contribution)
- [ ] Decide granularity (unstructured vs structured)
- [ ] Implement proper weight initialization for new capacity
- [ ] Handle downstream dimension changes after growth/pruning
- [ ] Consider optimizer state (momentum for new params?)
- [ ] Add observability (log structural changes)

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Growing too fast | Instability, wasted capacity | Use plateau detection, require improvement before next growth |
| Growing too slow | Underfitting, slow progress | Lower patience threshold, consider learned triggers |
| Pruning too aggressively | Lose important capacity | Use gradual pruning, verify with holdout |
| Ignoring optimizer | New params have no momentum | Either reset optimizer or initialize momentum for new params |
| Not updating downstream | Dimension mismatch | When widening layer N, update layer N+1 input |
| No cooldown | Thrashing between grow/prune | Implement embargo or minimum stabilization period |
