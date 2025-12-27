# Progressive Training Strategies

## Overview

Progressive training expands capability in stages - adding capacity gradually rather than all at once. This is curriculum learning applied to architecture, not just data.

**Core insight:** Sudden architectural changes destabilize training. Gradual expansion with proper warmup and cooldown maintains stability while growing capability.

---

## Progressive Training Philosophy

### Traditional Curriculum (Data)

Sort training data from easy to hard:
- Start with simple examples
- Gradually introduce complexity
- Helps optimization find good basins

### Architecture Curriculum

Sort model capacity from small to large:
- Start with small model
- Grow capacity as needed
- Each stage builds on previous learning

### Combined Approach

Both data and architecture curriculum together:
- Small model on easy data
- Grow model as data gets harder
- Maximum stability, gradual capability

```python
class CombinedCurriculum:
    def __init__(self, data_scheduler, arch_scheduler):
        self.data_scheduler = data_scheduler
        self.arch_scheduler = arch_scheduler

    def step(self, epoch, model, metrics):
        # Update data difficulty
        self.data_scheduler.step(epoch, metrics)

        # Update architecture if data curriculum hits plateau
        if self.data_scheduler.at_max_difficulty():
            if metrics.get('loss_plateaued'):
                self.arch_scheduler.grow(model)
                self.data_scheduler.reset()  # Start data curriculum again
```

---

## Staged Capacity Expansion

### Start Small Strategy

Begin with minimal architecture, grow only when needed.

```python
class StartSmallTrainer:
    def __init__(self, initial_model, growth_config):
        self.model = initial_model
        self.config = growth_config
        self.stage = 0
        self.plateau_detector = PlateauDetector(
            patience=self.config['patience'],
            threshold=self.config['improvement_threshold']
        )

    def train_epoch(self, data, optimizer):
        loss = train_one_epoch(self.model, data, optimizer)

        if self.plateau_detector.update(loss):
            self.grow_model()
            self.plateau_detector.reset()

        return loss

    def grow_model(self):
        """Add capacity according to growth schedule"""
        self.stage += 1
        growth_spec = self.config['stages'].get(self.stage)
        if growth_spec:
            self.model = apply_growth(self.model, growth_spec)
            print(f"Grew to stage {self.stage}: {growth_spec}")
```

**Benefits:**
- Faster early iterations (small model = fast epochs)
- Clearer learning signal (fewer parameters to confuse)
- Natural stopping (stop growing when no more improvement)

### Grow on Plateau Strategy

Detect when current capacity is exhausted.

```python
class PlateauDetector:
    def __init__(self, patience=10, threshold=0.001, min_delta=0.0):
        self.patience = patience
        self.threshold = threshold
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.history = []

    def update(self, loss):
        """Returns True if plateau detected"""
        self.history.append(loss)

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            return False

        self.wait += 1

        # Check if improvement rate is below threshold
        if len(self.history) >= self.patience:
            recent = self.history[-self.patience:]
            improvement_rate = (recent[0] - recent[-1]) / self.patience
            if improvement_rate < self.threshold:
                return True

        return self.wait >= self.patience

    def reset(self):
        self.wait = 0
        # Don't reset best_loss - we want improvement over all time
```

### Transfer Knowledge Across Stages

Don't start from scratch when growing.

```python
def grow_with_transfer(model, new_capacity_spec):
    """Grow model while preserving learned features"""

    # Save current model state
    old_state = model.state_dict()
    old_config = model.config

    # Create larger model
    new_config = merge_configs(old_config, new_capacity_spec)
    new_model = create_model(new_config)

    # Transfer weights where dimensions match
    new_state = new_model.state_dict()
    for key in old_state:
        if key in new_state:
            old_tensor = old_state[key]
            new_tensor = new_state[key]

            if old_tensor.shape == new_tensor.shape:
                new_state[key] = old_tensor
            else:
                # Partial transfer for grown dimensions
                new_state[key] = transfer_partial(old_tensor, new_tensor)

    new_model.load_state_dict(new_state)
    return new_model

def transfer_partial(old_tensor, new_tensor):
    """Transfer overlapping dimensions"""
    result = new_tensor.clone()
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape))
    result[slices] = old_tensor[slices]
    return result
```

---

## Module Warmup Patterns

When adding new capacity, it shouldn't disrupt existing training.

### Zero-Init Warmup

New module starts with zero output, gradually increases.

```python
class ZeroInitWarmup(nn.Module):
    def __init__(self, module, warmup_steps=1000):
        super().__init__()
        self.module = module
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Initialize output layer to zero
        self._zero_init_output()

    def _zero_init_output(self):
        """Zero the final layer so module outputs nothing initially"""
        for name, param in self.module.named_parameters():
            if 'output' in name or 'final' in name or 'head' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out = self.module(x)

        # Scale output by warmup progress
        if self.training and self.current_step < self.warmup_steps:
            scale = self.current_step / self.warmup_steps
            out = out * scale
            self.current_step += 1

        return out
```

### Learning Rate Warmup

New module gets lower learning rate initially.

```python
class ModuleLRWarmup:
    def __init__(self, optimizer, module_params, base_lr, warmup_steps=1000):
        self.optimizer = optimizer
        self.module_param_ids = {id(p) for p in module_params}
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Add module params to optimizer with zero LR
        self.optimizer.add_param_group({
            'params': list(module_params),
            'lr': 0.0
        })
        self.module_group_idx = len(self.optimizer.param_groups) - 1

    def step(self):
        """Call after each training step"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            new_lr = self.base_lr * progress
            self.optimizer.param_groups[self.module_group_idx]['lr'] = new_lr
```

### Alpha Ramp Warmup

Blend new module output from 0 to 1.

```python
class AlphaRampWarmup(nn.Module):
    def __init__(self, host_module, new_module, warmup_steps=1000, schedule='linear'):
        super().__init__()
        self.host = host_module
        self.new = new_module
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.current_step = 0

    def get_alpha(self):
        if self.current_step >= self.warmup_steps:
            return 1.0

        progress = self.current_step / self.warmup_steps

        if self.schedule == 'linear':
            return progress
        elif self.schedule == 'cosine':
            return 0.5 * (1 - math.cos(math.pi * progress))
        elif self.schedule == 'exponential':
            return 1 - math.exp(-5 * progress)

    def forward(self, x):
        host_out = self.host(x)

        if self.training:
            alpha = self.get_alpha()
            self.current_step += 1

            if alpha < 1.0:
                new_out = self.new(x.detach())  # Isolate during warmup
                return host_out + alpha * new_out
            else:
                # Full integration after warmup
                new_out = self.new(x)
                return host_out + new_out
        else:
            return host_out + self.new(x)
```

### Frozen Host Warmup

Train new module in complete isolation first.

```python
class FrozenHostWarmup:
    def __init__(self, host, new_module, isolation_epochs=5):
        self.host = host
        self.new = new_module
        self.isolation_epochs = isolation_epochs
        self.current_epoch = 0

    def train_epoch(self, data, host_optimizer, new_optimizer):
        if self.current_epoch < self.isolation_epochs:
            # Phase 1: Only train new module, host frozen
            self._freeze(self.host)
            loss = train_one_epoch(self.new, data, new_optimizer)
            self._unfreeze(self.host)
        else:
            # Phase 2: Joint training
            loss = train_one_epoch(
                CombinedModel(self.host, self.new),
                data,
                [host_optimizer, new_optimizer]
            )

        self.current_epoch += 1
        return loss

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
```

---

## Cooldown & Stabilization

After adding capacity, allow the system to stabilize.

### Post-Integration Settling Period

Reduce learning rate after integration.

```python
class SettlingPeriod:
    def __init__(self, optimizer, settle_epochs=5, lr_factor=0.1):
        self.optimizer = optimizer
        self.settle_epochs = settle_epochs
        self.lr_factor = lr_factor
        self.original_lrs = None
        self.epochs_settled = 0

    def begin_settling(self):
        """Call after module integration"""
        self.original_lrs = [g['lr'] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g['lr'] *= self.lr_factor
        self.epochs_settled = 0

    def step(self):
        """Call each epoch during settling"""
        self.epochs_settled += 1
        if self.epochs_settled >= self.settle_epochs:
            self.end_settling()
            return True  # Settling complete
        return False

    def end_settling(self):
        """Restore original learning rates"""
        if self.original_lrs:
            for g, lr in zip(self.optimizer.param_groups, self.original_lrs):
                g['lr'] = lr
            self.original_lrs = None
```

### Contribution Monitoring

Watch for regression after integration.

```python
class ContributionMonitor:
    def __init__(self, model, eval_fn, regression_threshold=0.02):
        self.model = model
        self.eval_fn = eval_fn
        self.regression_threshold = regression_threshold
        self.baseline_metric = None
        self.integrated_modules = []

    def record_baseline(self):
        """Call before integration"""
        self.baseline_metric = self.eval_fn(self.model)

    def add_integrated_module(self, module_id):
        self.integrated_modules.append({
            'id': module_id,
            'baseline': self.baseline_metric,
            'post_integration': None
        })

    def check_regression(self):
        """Returns list of regressing modules"""
        current = self.eval_fn(self.model)
        regressing = []

        for module_info in self.integrated_modules:
            baseline = module_info['baseline']
            if current < baseline * (1 - self.regression_threshold):
                regressing.append(module_info['id'])

        return regressing
```

### Consolidation Epochs

Period of no structural changes, only weight refinement.

```python
class ConsolidationPeriod:
    def __init__(self, growth_manager, min_epochs=10):
        self.growth_manager = growth_manager
        self.min_epochs = min_epochs
        self.epochs_since_change = 0
        self.in_consolidation = False

    def begin_consolidation(self):
        """Start consolidation period"""
        self.in_consolidation = True
        self.epochs_since_change = 0
        self.growth_manager.disable()

    def step(self):
        if self.in_consolidation:
            self.epochs_since_change += 1
            if self.epochs_since_change >= self.min_epochs:
                self.end_consolidation()
                return True
        return False

    def end_consolidation(self):
        self.in_consolidation = False
        self.growth_manager.enable()
```

---

## Multi-Stage Training Schedules

### Sequential Stages

Complete one stage before starting next.

```python
class SequentialStages:
    def __init__(self, stages):
        """
        stages: list of dicts with 'condition', 'action', 'consolidation'
        """
        self.stages = stages
        self.current_stage = 0

    def step(self, epoch, metrics, model):
        if self.current_stage >= len(self.stages):
            return  # All stages complete

        stage = self.stages[self.current_stage]

        if stage['condition'](metrics):
            # Execute stage action
            stage['action'](model)

            # Enter consolidation
            if 'consolidation' in stage:
                stage['consolidation'].begin()

            self.current_stage += 1

# Example usage
stages = [
    {
        'condition': lambda m: m['accuracy'] > 0.7,
        'action': lambda model: add_attention_layer(model),
        'consolidation': ConsolidationPeriod(growth_mgr, min_epochs=5)
    },
    {
        'condition': lambda m: m['accuracy'] > 0.85,
        'action': lambda model: widen_layers(model, factor=1.5),
        'consolidation': ConsolidationPeriod(growth_mgr, min_epochs=10)
    }
]
```

### Overlapping Stages

Start new stage while previous is still consolidating.

```python
class OverlappingStages:
    def __init__(self):
        self.active_stages = []
        self.pending_stages = []

    def add_stage(self, stage):
        self.pending_stages.append(stage)

    def step(self, epoch, metrics, model):
        # Check if any pending stage should start
        for stage in self.pending_stages[:]:
            if stage['trigger'](metrics):
                self.pending_stages.remove(stage)
                stage['start'](model)
                stage['warmup'] = stage.get('warmup_epochs', 0)
                self.active_stages.append(stage)

        # Progress active stages
        for stage in self.active_stages[:]:
            if stage['warmup'] > 0:
                stage['warmup'] -= 1
            else:
                stage['consolidate'](model)
                if stage.get('complete', lambda m: False)(metrics):
                    self.active_stages.remove(stage)
```

### Budget-Aware Scheduling

Grow until parameter budget is exhausted.

```python
class BudgetAwareScheduler:
    def __init__(self, model, param_budget, growth_options):
        self.model = model
        self.param_budget = param_budget
        self.growth_options = growth_options  # List of (growth_fn, param_cost)

    def current_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def headroom(self):
        return self.param_budget - self.current_params()

    def step(self, metrics):
        if not self._should_grow(metrics):
            return

        # Find largest growth that fits in budget
        affordable = [(fn, cost) for fn, cost in self.growth_options
                      if cost <= self.headroom()]

        if affordable:
            # Pick growth with best expected value (or just largest)
            best_fn, best_cost = max(affordable, key=lambda x: x[1])
            best_fn(self.model)
            print(f"Grew by {best_cost} params, {self.headroom()} remaining")

    def _should_grow(self, metrics):
        return metrics.get('loss_plateaued', False) and self.headroom() > 0
```

---

## Knowledge Transfer Between Stages

### Weight Inheritance

Copy weights from smaller to larger architecture.

```python
def inherit_weights(small_model, large_model):
    """
    Copy weights from small model into corresponding positions in large model.
    Assumes large model is a superset of small model architecture.
    """
    small_state = small_model.state_dict()
    large_state = large_model.state_dict()

    for key in small_state:
        if key in large_state:
            small_tensor = small_state[key]
            large_tensor = large_state[key]

            if small_tensor.shape == large_tensor.shape:
                # Exact match: direct copy
                large_state[key] = small_tensor
            elif all(s <= l for s, l in zip(small_tensor.shape, large_tensor.shape)):
                # Small fits inside large: copy to upper-left corner
                slices = tuple(slice(0, s) for s in small_tensor.shape)
                large_state[key][slices] = small_tensor
            else:
                print(f"Cannot inherit {key}: shape mismatch {small_tensor.shape} vs {large_tensor.shape}")

    large_model.load_state_dict(large_state)
    return large_model
```

### Distillation Between Stages

Use previous stage as teacher for current stage.

```python
class StageDistillation:
    def __init__(self, temperature=2.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha  # Balance between hard and soft labels

    def create_teacher(self, model):
        """Freeze and save model as teacher"""
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def distillation_loss(self, student_logits, teacher_logits, hard_labels):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard targets
        hard_loss = F.cross_entropy(student_logits, hard_labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

class ProgressiveDistillation:
    def __init__(self):
        self.teachers = []  # Stack of previous stage models

    def complete_stage(self, model):
        """Save current model as teacher for next stage"""
        teacher = StageDistillation().create_teacher(model)
        self.teachers.append(teacher)

    def get_distillation_targets(self, x):
        """Get soft targets from all previous teachers"""
        targets = []
        for teacher in self.teachers:
            with torch.no_grad():
                targets.append(teacher(x))
        return targets
```

### Feature Reuse

Share backbone, add specialized heads per stage.

```python
class SharedBackboneProgressive(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self.current_stage = 0

    def add_stage_head(self, stage_name, head):
        """Add head for new stage, keeping backbone shared"""
        self.heads[stage_name] = head
        self.current_stage += 1

        # Optionally freeze backbone when adding first head
        if len(self.heads) == 1:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, stage=None):
        features = self.backbone(x)

        if stage is None:
            stage = list(self.heads.keys())[-1]  # Use latest head

        return self.heads[stage](features)
```

---

## Failure Modes

### Growing Too Fast

**Symptoms:**
- Training loss becomes unstable after growth
- Accuracy drops significantly after adding capacity
- Gradient norms spike

**Solutions:**
- Longer plateau detection (higher patience)
- More aggressive warmup
- Smaller growth increments

```python
# Symptom detection
def detect_growth_instability(metrics_history, growth_step):
    pre_growth = metrics_history[growth_step - 10:growth_step]
    post_growth = metrics_history[growth_step:growth_step + 10]

    loss_spike = max(post_growth) > 1.5 * max(pre_growth)
    variance_spike = np.var(post_growth) > 3 * np.var(pre_growth)

    return loss_spike or variance_spike
```

### Growing Too Slow

**Symptoms:**
- Loss plateaus for extended periods
- Model underperforms despite having capacity budget
- Training takes longer than necessary

**Solutions:**
- Lower plateau patience
- More sensitive improvement threshold
- Proactive growth (grow before plateau)

### Stage Transition Shock

**Symptoms:**
- Sharp performance drop at stage transition
- Takes many epochs to recover previous performance
- Sometimes never recovers

**Solutions:**
- Longer warmup periods
- Learning rate cooldown at transition
- Distillation from previous stage
- Smaller stage transitions

```python
class TransitionShockPrevention:
    def __init__(self, lr_reduction=0.1, recovery_threshold=0.95):
        self.lr_reduction = lr_reduction
        self.recovery_threshold = recovery_threshold
        self.pre_transition_metric = None

    def before_transition(self, model, metrics):
        self.pre_transition_metric = metrics['accuracy']

    def after_transition(self, optimizer):
        # Reduce learning rate
        for g in optimizer.param_groups:
            g['lr'] *= self.lr_reduction

    def check_recovery(self, metrics):
        current = metrics['accuracy']
        target = self.pre_transition_metric * self.recovery_threshold
        return current >= target
```

---

## Implementation Checklist

When implementing progressive training:

- [ ] Define growth triggers (plateau detection, budget, schedule)
- [ ] Choose warmup strategy (zero-init, LR warmup, alpha ramp, frozen host)
- [ ] Implement knowledge transfer (weight inheritance, distillation)
- [ ] Plan consolidation periods after each stage
- [ ] Monitor for regression after growth
- [ ] Handle stage transition (LR reduction, settling period)
- [ ] Define stopping criteria (budget exhausted, no improvement)
- [ ] Log all stage transitions and metrics
