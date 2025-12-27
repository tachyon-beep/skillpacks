# ML Lifecycle Orchestration

## Overview

Lifecycle orchestration is the "when" of dynamic architectures. While other skills cover "how" to grow, prune, and compose modules, this skill covers when to take those actions and how to manage the state transitions safely.

**Core insight:** Training is not just gradient descent - it's a controlled process with states, transitions, and decision points. Explicit state machines prevent premature actions and enable systematic debugging.

---

## Why Lifecycle Management

### Training as Controlled Process

Naive approach: "Add capacity when loss plateaus, remove when contribution is low."

Problem: Without explicit state tracking, you get:
- Premature integration (module not ready)
- Delayed pruning (keeping dead weight)
- Thrashing (grow-prune-grow-prune cycles)
- No recovery (failure cascades)

Solution: Explicit states, guarded transitions, and recovery mechanisms.

### Benefits of State Machines

1. **Clarity**: Know exactly what state each module is in
2. **Guards**: Transitions require explicit conditions
3. **Observability**: Log state changes, debug systematically
4. **Recovery**: Define what happens on failure
5. **Automation**: State machines can be controlled by heuristics or RL

---

## State Machine Fundamentals

### States as Module Conditions

Each module in a dynamic system has a lifecycle state:

```python
from enum import Enum, auto

class ModuleState(Enum):
    # Creation states
    DORMANT = auto()      # Slot empty, available for new module
    INSTANTIATED = auto() # Module created, not yet training

    # Active states
    TRAINING = auto()     # Module receiving gradients, learning
    BLENDING = auto()     # Module output being alpha-blended into host
    HOLDING = auto()      # Module at full contribution, under observation

    # Terminal states (success)
    INTEGRATED = auto()   # Module permanently part of host

    # Terminal states (failure)
    PRUNED = auto()       # Module removed due to poor performance
    EMBARGOED = auto()    # Slot on cooldown, cannot be reused yet

class ModuleLifecycle:
    def __init__(self, module_id):
        self.module_id = module_id
        self.state = ModuleState.DORMANT
        self.state_history = []
        self.metrics = {}

    def transition(self, new_state, reason=None):
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            'from': old_state,
            'to': new_state,
            'reason': reason,
            'timestamp': time.time()
        })
```

### Transitions as Guarded Actions

Transitions only occur when guards (conditions) are satisfied:

```python
class TransitionGuard:
    def __init__(self, name, condition_fn, description):
        self.name = name
        self.condition = condition_fn
        self.description = description

    def check(self, lifecycle, metrics):
        result = self.condition(lifecycle, metrics)
        return result, self.description if not result else None

class StateMachine:
    def __init__(self):
        self.transitions = {}  # (from_state, to_state) -> list of guards

    def add_transition(self, from_state, to_state, guards):
        self.transitions[(from_state, to_state)] = guards

    def can_transition(self, lifecycle, to_state, metrics):
        key = (lifecycle.state, to_state)
        if key not in self.transitions:
            return False, f"No transition defined from {lifecycle.state} to {to_state}"

        for guard in self.transitions[key]:
            passed, reason = guard.check(lifecycle, metrics)
            if not passed:
                return False, reason

        return True, None

    def transition(self, lifecycle, to_state, metrics):
        can, reason = self.can_transition(lifecycle, to_state, metrics)
        if can:
            lifecycle.transition(to_state, reason="Guards passed")
            return True
        return False
```

### Terminal States

States that end the module's active lifecycle:

```python
TERMINAL_STATES = {
    ModuleState.INTEGRATED,  # Success: module is now permanent
    ModuleState.PRUNED,      # Failure: module was removed
    ModuleState.EMBARGOED    # Failure + cooldown
}

def is_terminal(state):
    return state in TERMINAL_STATES

def can_recycle(state, cooldown_elapsed):
    """Check if module/slot can be reused"""
    if state == ModuleState.INTEGRATED:
        return False  # Integrated modules don't recycle
    if state == ModuleState.EMBARGOED:
        return cooldown_elapsed
    if state == ModuleState.PRUNED:
        return True  # Can immediately reuse after pruning
    return False
```

---

## Gate Design Patterns

Gates are checkpoints that must pass before transitions occur.

### Structural Gates

Prerequisites about system state, not module performance.

```python
class StructuralGates:
    @staticmethod
    def slot_available(lifecycle, metrics):
        """Is there an empty slot for growth?"""
        return metrics.get('available_slots', 0) > 0

    @staticmethod
    def budget_permits(lifecycle, metrics):
        """Is there parameter budget for new module?"""
        current = metrics.get('current_params', 0)
        budget = metrics.get('param_budget', float('inf'))
        proposed = metrics.get('proposed_params', 0)
        return current + proposed <= budget

    @staticmethod
    def no_active_transitions(lifecycle, metrics):
        """No other modules mid-transition (stability)"""
        return metrics.get('modules_in_transition', 0) == 0
```

### Performance Gates

Metrics about module or system performance.

```python
class PerformanceGates:
    @staticmethod
    def minimum_epochs_trained(lifecycle, metrics, min_epochs=5):
        """Module has trained enough to evaluate"""
        return metrics.get('epochs_trained', 0) >= min_epochs

    @staticmethod
    def loss_improved(lifecycle, metrics, threshold=0.01):
        """Training loss improved since module was added"""
        baseline = metrics.get('loss_at_instantiation', float('inf'))
        current = metrics.get('current_loss', float('inf'))
        return current < baseline * (1 - threshold)

    @staticmethod
    def no_regression(lifecycle, metrics, tolerance=0.02):
        """Host performance didn't degrade"""
        baseline = metrics.get('host_accuracy_at_blend', 1.0)
        current = metrics.get('current_accuracy', 0.0)
        return current >= baseline * (1 - tolerance)
```

### Stability Gates

Performance must be consistent, not just good once.

```python
class StabilityGates:
    @staticmethod
    def consistent_improvement(lifecycle, metrics, window=10, variance_threshold=0.1):
        """Improvement is stable, not fluctuating"""
        history = metrics.get('improvement_history', [])
        if len(history) < window:
            return False
        recent = history[-window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance < variance_threshold and mean > 0

    @staticmethod
    def gradient_health(lifecycle, metrics, max_grad_norm=10.0, min_grad_norm=1e-6):
        """Gradients are healthy (not exploding or vanishing)"""
        grad_norm = metrics.get('module_grad_norm', 0.0)
        return min_grad_norm < grad_norm < max_grad_norm
```

### Contribution Gates

Module is actually helping, not just surviving.

```python
class ContributionGates:
    @staticmethod
    def positive_contribution(lifecycle, metrics, threshold=0.0):
        """Module contributes positively (counterfactual test)"""
        contribution = metrics.get('module_contribution', 0.0)
        return contribution > threshold

    @staticmethod
    def contribution_above_cost(lifecycle, metrics):
        """Contribution exceeds compute/memory cost"""
        contribution = metrics.get('module_contribution', 0.0)
        cost = metrics.get('module_cost', 0.0)
        return contribution > cost

    @staticmethod
    def unique_contribution(lifecycle, metrics, overlap_threshold=0.5):
        """Module provides something other modules don't"""
        overlap = metrics.get('contribution_overlap', 0.0)
        return overlap < overlap_threshold
```

### Permissive vs Strict Modes

```python
class GateMode(Enum):
    STRICT = auto()    # All gates must pass
    PERMISSIVE = auto() # Only structural gates required

class AdaptiveGateChecker:
    def __init__(self, mode=GateMode.STRICT):
        self.mode = mode
        self.structural_gates = [...]
        self.performance_gates = [...]
        self.stability_gates = [...]

    def check_all(self, lifecycle, metrics):
        # Structural always required
        for gate in self.structural_gates:
            if not gate(lifecycle, metrics):
                return False, "Structural gate failed"

        if self.mode == GateMode.PERMISSIVE:
            return True, None  # Skip performance/stability

        # Performance and stability in strict mode
        for gate in self.performance_gates:
            if not gate(lifecycle, metrics):
                return False, "Performance gate failed"

        for gate in self.stability_gates:
            if not gate(lifecycle, metrics):
                return False, "Stability gate failed"

        return True, None
```

**When to use each mode:**

| Mode | Use When |
|------|----------|
| Strict | Production, need reliable decisions |
| Permissive | Letting RL controller learn thresholds |

---

## Transition Trigger Types

### Metric-Based Triggers

Transition when metrics cross thresholds.

```python
class MetricTrigger:
    def __init__(self, metric_name, comparator, threshold):
        self.metric_name = metric_name
        self.comparator = comparator  # 'gt', 'lt', 'eq'
        self.threshold = threshold

    def check(self, metrics):
        value = metrics.get(self.metric_name)
        if value is None:
            return False

        if self.comparator == 'gt':
            return value > self.threshold
        elif self.comparator == 'lt':
            return value < self.threshold
        elif self.comparator == 'eq':
            return abs(value - self.threshold) < 1e-6

# Usage
loss_plateau = MetricTrigger('loss_delta', 'lt', 0.001)
accuracy_improved = MetricTrigger('accuracy', 'gt', 0.85)
```

### Time-Based Triggers

Transition after minimum/maximum time in state.

```python
class TimeTrigger:
    def __init__(self, min_steps=None, max_steps=None, min_epochs=None, max_epochs=None):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

    def check(self, lifecycle, metrics):
        steps_in_state = metrics.get('steps_in_current_state', 0)
        epochs_in_state = metrics.get('epochs_in_current_state', 0)

        # Minimum time requirements
        if self.min_steps and steps_in_state < self.min_steps:
            return False, "Minimum steps not reached"
        if self.min_epochs and epochs_in_state < self.min_epochs:
            return False, "Minimum epochs not reached"

        # Maximum time (force transition)
        if self.max_steps and steps_in_state >= self.max_steps:
            return True, "Maximum steps reached"
        if self.max_epochs and epochs_in_state >= self.max_epochs:
            return True, "Maximum epochs reached"

        return None, "Within time bounds"  # Neither forced nor blocked
```

### Budget-Based Triggers

Transition based on resource constraints.

```python
class BudgetTrigger:
    def __init__(self, param_budget, flops_budget=None, memory_budget=None):
        self.param_budget = param_budget
        self.flops_budget = flops_budget
        self.memory_budget = memory_budget

    def should_prune(self, metrics):
        """Check if we must prune due to budget"""
        current_params = metrics.get('total_params', 0)
        return current_params > self.param_budget

    def can_grow(self, metrics, proposed_params):
        """Check if growth is allowed"""
        current = metrics.get('total_params', 0)
        return current + proposed_params <= self.param_budget
```

### Controller-Driven Triggers

External controller (heuristic or RL) decides transitions.

```python
class ControllerTrigger:
    def __init__(self, controller):
        self.controller = controller

    def get_action(self, state, metrics):
        """Controller decides what transition to attempt"""
        observation = self._build_observation(state, metrics)
        action = self.controller.act(observation)
        return action  # e.g., 'advance', 'prune', 'hold'

    def _build_observation(self, state, metrics):
        return {
            'state': state.value,
            'loss': metrics.get('current_loss'),
            'contribution': metrics.get('module_contribution'),
            'epochs_in_state': metrics.get('epochs_in_current_state'),
            # ... other relevant features
        }
```

---

## Rollback & Recovery

### Cooldown Patterns

After failure, wait before retrying.

```python
class CooldownManager:
    def __init__(self, base_cooldown=100, exponential_backoff=True):
        self.base_cooldown = base_cooldown
        self.exponential_backoff = exponential_backoff
        self.failure_counts = {}  # slot_id -> count
        self.embargo_until = {}   # slot_id -> step

    def start_embargo(self, slot_id, current_step):
        count = self.failure_counts.get(slot_id, 0) + 1
        self.failure_counts[slot_id] = count

        if self.exponential_backoff:
            cooldown = self.base_cooldown * (2 ** (count - 1))
        else:
            cooldown = self.base_cooldown

        self.embargo_until[slot_id] = current_step + cooldown

    def is_embargoed(self, slot_id, current_step):
        if slot_id not in self.embargo_until:
            return False
        return current_step < self.embargo_until[slot_id]

    def reset_failures(self, slot_id):
        """Call on successful integration"""
        self.failure_counts[slot_id] = 0
        self.embargo_until.pop(slot_id, None)
```

### Checkpoint-Based Rollback

Restore previous state on failure.

```python
class CheckpointRollback:
    def __init__(self, model):
        self.model = model
        self.checkpoints = {}

    def save_checkpoint(self, name):
        """Save before risky operation"""
        self.checkpoints[name] = {
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'timestamp': time.time()
        }

    def rollback(self, name):
        """Restore to checkpoint"""
        if name not in self.checkpoints:
            raise ValueError(f"No checkpoint named {name}")
        self.model.load_state_dict(self.checkpoints[name]['state_dict'])

    def discard(self, name):
        """Discard checkpoint after successful operation"""
        self.checkpoints.pop(name, None)

# Usage in lifecycle
checkpoint.save_checkpoint('before_integration')
try:
    integrate_module(module)
    if not verify_integration():
        checkpoint.rollback('before_integration')
    else:
        checkpoint.discard('before_integration')
except Exception:
    checkpoint.rollback('before_integration')
```

### Hysteresis (Debounce State Thrashing)

Prevent rapid state changes.

```python
class HysteresisGuard:
    def __init__(self, min_time_in_state=50, max_transitions_per_window=3, window_size=100):
        self.min_time_in_state = min_time_in_state
        self.max_transitions = max_transitions_per_window
        self.window_size = window_size
        self.transition_history = []

    def can_transition(self, lifecycle, current_step):
        # Minimum time in current state
        time_in_state = current_step - lifecycle.state_history[-1]['timestamp']
        if time_in_state < self.min_time_in_state:
            return False, "Too soon since last transition"

        # Maximum transitions in window
        recent = [t for t in self.transition_history
                  if current_step - t < self.window_size]
        if len(recent) >= self.max_transitions:
            return False, "Too many transitions in window"

        return True, None

    def record_transition(self, step):
        self.transition_history.append(step)
        # Prune old entries
        self.transition_history = [t for t in self.transition_history
                                   if step - t < self.window_size * 2]
```

---

## Controller Patterns

### Heuristic Controller

Rule-based decisions, predictable behavior.

```python
class HeuristicController:
    def __init__(self, config):
        self.config = config

    def decide(self, state, metrics):
        """Return action based on rules"""

        if state == ModuleState.TRAINING:
            epochs = metrics.get('epochs_in_state', 0)
            loss_delta = metrics.get('loss_delta', 0)

            # Advance if trained enough and improving
            if epochs >= self.config['min_training_epochs']:
                if loss_delta < -self.config['improvement_threshold']:
                    return 'advance'
                elif loss_delta > self.config['regression_threshold']:
                    return 'prune'
            return 'hold'

        elif state == ModuleState.BLENDING:
            contribution = metrics.get('contribution', 0)
            regression = metrics.get('host_regression', 0)

            if regression > self.config['max_regression']:
                return 'prune'
            elif contribution > self.config['min_contribution']:
                return 'advance'
            return 'hold'

        # ... other states

        return 'hold'  # Default: do nothing
```

### Learned Controller (RL)

Train a policy to make decisions.

```python
class RLController:
    def __init__(self, observation_dim, action_space, policy_network):
        self.policy = policy_network
        self.action_space = action_space  # ['hold', 'advance', 'prune', 'grow']

    def act(self, observation, deterministic=False):
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            if deterministic:
                action_idx = logits.argmax()
            else:
                action_idx = torch.distributions.Categorical(logits=logits).sample()
        return self.action_space[action_idx]

    def get_reward(self, metrics, action_taken):
        """Reward function for RL training"""
        reward = 0

        # Reward improvement
        if metrics.get('loss_improved'):
            reward += 1.0

        # Penalize regression
        if metrics.get('host_regression', 0) > 0:
            reward -= 2.0

        # Penalize budget overrun
        if metrics.get('over_budget'):
            reward -= 1.0

        # Reward successful integration
        if action_taken == 'advance' and metrics.get('integration_success'):
            reward += 5.0

        return reward
```

### Hybrid Controller

Heuristic baseline with learned refinements.

```python
class HybridController:
    def __init__(self, heuristic, learned, blend_alpha=0.5):
        self.heuristic = heuristic
        self.learned = learned
        self.blend_alpha = blend_alpha

    def decide(self, state, metrics):
        heuristic_action = self.heuristic.decide(state, metrics)
        learned_action = self.learned.act(self._to_obs(state, metrics))

        # Use heuristic as safety net
        if heuristic_action == 'prune' and metrics.get('regression', 0) > 0.1:
            return 'prune'  # Override learned if obvious failure

        # Blend based on confidence
        learned_confidence = self.learned.get_confidence()
        if learned_confidence > 0.8:
            return learned_action
        elif learned_confidence < 0.3:
            return heuristic_action
        else:
            # Weighted random choice
            if random.random() < self.blend_alpha:
                return learned_action
            return heuristic_action
```

---

## Observability

### State Transition Events

```python
class LifecycleLogger:
    def __init__(self):
        self.events = []

    def log_transition(self, module_id, from_state, to_state, reason, metrics):
        event = {
            'timestamp': time.time(),
            'module_id': module_id,
            'from_state': from_state.name,
            'to_state': to_state.name,
            'reason': reason,
            'metrics_snapshot': dict(metrics)
        }
        self.events.append(event)
        print(f"[LIFECYCLE] {module_id}: {from_state.name} -> {to_state.name} ({reason})")

    def get_history(self, module_id=None):
        if module_id:
            return [e for e in self.events if e['module_id'] == module_id]
        return self.events
```

### Gate Pass/Fail Telemetry

```python
class GateTelemetry:
    def __init__(self):
        self.gate_results = []

    def record_check(self, gate_name, passed, metrics, reason=None):
        self.gate_results.append({
            'timestamp': time.time(),
            'gate': gate_name,
            'passed': passed,
            'reason': reason,
            'metrics': dict(metrics)
        })

    def get_failure_reasons(self, gate_name=None):
        failures = [r for r in self.gate_results if not r['passed']]
        if gate_name:
            failures = [r for r in failures if r['gate'] == gate_name]
        return failures

    def get_pass_rate(self, gate_name):
        relevant = [r for r in self.gate_results if r['gate'] == gate_name]
        if not relevant:
            return None
        return sum(r['passed'] for r in relevant) / len(relevant)
```

### Time-in-State Metrics

```python
class StateTimeTracker:
    def __init__(self):
        self.state_times = {}  # module_id -> {state: total_time}
        self.current_state_start = {}  # module_id -> start_time

    def enter_state(self, module_id, state):
        self.current_state_start[module_id] = time.time()

    def exit_state(self, module_id, state):
        if module_id in self.current_state_start:
            duration = time.time() - self.current_state_start[module_id]
            if module_id not in self.state_times:
                self.state_times[module_id] = {}
            if state not in self.state_times[module_id]:
                self.state_times[module_id][state] = 0
            self.state_times[module_id][state] += duration

    def get_average_time(self, state):
        times = [st.get(state, 0) for st in self.state_times.values()]
        return sum(times) / len(times) if times else 0
```

---

## Implementation Checklist

When implementing lifecycle orchestration:

- [ ] Define all states (including terminal states)
- [ ] Define valid transitions between states
- [ ] Implement guards for each transition
- [ ] Choose trigger types (metric, time, budget, controller)
- [ ] Implement cooldown/embargo for failures
- [ ] Add checkpoint-based rollback for risky transitions
- [ ] Implement hysteresis to prevent thrashing
- [ ] Add logging for all state transitions
- [ ] Track time-in-state for debugging
- [ ] Test recovery paths (what happens on failure?)
