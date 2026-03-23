---
description: Design a lifecycle state machine for managing neural module growth, training, and integration
allowed-tools: ["Read", "Glob", "Grep", "AskUserQuestion"]
---

# Design Module Lifecycle

Help design a state machine for managing neural module lifecycles - the states, transitions, gates, and controllers that orchestrate dynamic architecture.

## Design Process

### Step 1: Gather Requirements

Ask about the use case:

1. **Module types**: What kinds of modules will have lifecycles?
   - Seed modules (grow and graft)
   - Expert modules (MoE routing)
   - Prunable components (dynamic sparsity)

2. **Lifecycle scope**: What phases matter?
   - Creation/instantiation
   - Isolated training
   - Integration/blending
   - Evaluation/holding
   - Permanent integration
   - Removal/recycling

3. **Control strategy**: Who decides transitions?
   - Heuristic rules
   - Learned policy (RL)
   - Hybrid

4. **Failure handling**: What happens when things go wrong?
   - Rollback capability
   - Cooldown/embargo
   - Retry limits

### Step 2: Explore Existing Code

Look for existing lifecycle patterns:

```
Glob: **/state*.py, **/lifecycle*.py, **/stage*.py
Grep: "state|transition|gate|phase"
```

Identify constraints from current architecture:
- How are modules created?
- Where do they attach to host?
- How is training controlled?

### Step 3: Define States

Based on requirements, propose states:

```python
class ModuleState(Enum):
    # Early lifecycle
    DORMANT = auto()      # Slot empty
    INSTANTIATED = auto() # Module created

    # Active training
    TRAINING = auto()     # Learning in isolation
    BLENDING = auto()     # Gradually integrating

    # Evaluation
    HOLDING = auto()      # Full contribution, under observation

    # Terminal (success)
    INTEGRATED = auto()   # Permanently part of host

    # Terminal (failure)
    PRUNED = auto()       # Removed
    EMBARGOED = auto()    # On cooldown
```

### Step 4: Define Transitions

For each valid transition, specify:

1. **From state** → **To state**
2. **Required gates** (conditions that must pass)
3. **Trigger type** (what initiates check)
4. **Side effects** (what happens on transition)

Example transition table:

| From | To | Gates | Trigger | Side Effects |
|------|-----|-------|---------|--------------|
| DORMANT | INSTANTIATED | slot_available, budget_permits | controller_action | create_module, init_weights |
| TRAINING | BLENDING | min_epochs, loss_improved | metric_based | start_alpha_ramp |
| BLENDING | HOLDING | alpha_complete, no_regression | time_based | set_alpha_1.0 |
| HOLDING | INTEGRATED | stability_check, contribution_positive | metric_based | freeze_module |
| HOLDING | PRUNED | contribution_negative OR timeout | metric_based | remove_module |
| PRUNED | EMBARGOED | - | automatic | start_cooldown |
| EMBARGOED | DORMANT | cooldown_elapsed | time_based | reset_slot |

### Step 5: Define Gates

For each gate, specify:

1. **Name**
2. **Condition** (what must be true)
3. **Metrics needed** (what to measure)
4. **Threshold** (configurable value)

```python
gates = {
    'min_epochs': {
        'condition': 'epochs_in_state >= threshold',
        'metrics': ['epochs_in_state'],
        'default_threshold': 5
    },
    'loss_improved': {
        'condition': 'current_loss < baseline_loss * (1 - threshold)',
        'metrics': ['current_loss', 'baseline_loss'],
        'default_threshold': 0.01
    },
    'contribution_positive': {
        'condition': 'module_contribution > threshold',
        'metrics': ['module_contribution'],
        'default_threshold': 0.0
    }
}
```

### Step 6: Define Controller

Based on control strategy requirement:

**Heuristic Controller:**
```python
def decide(state, metrics):
    if state == TRAINING:
        if metrics['epochs'] >= 5 and metrics['loss_improved']:
            return 'advance'
        elif metrics['epochs'] >= 20 and not metrics['loss_improved']:
            return 'prune'
    return 'hold'
```

**Learned Controller (RL):**
```python
observation = [
    state_one_hot,
    normalized_loss,
    contribution_score,
    epochs_in_state,
    ...
]
action = policy(observation)  # 'hold', 'advance', 'prune'
```

### Step 7: Output Design Document

Provide complete lifecycle specification:

```markdown
## Lifecycle Design: [Name]

### States
[Table of states with descriptions]

### Transitions
[Table with from/to/gates/trigger/effects]

### Gates
[Table with name/condition/metrics/threshold]

### Controller
[Heuristic rules or RL observation/action space]

### Implementation Notes
[Key code patterns, integration points]
```

## Reference

For detailed patterns, load the ml-lifecycle-orchestration reference sheet:

```
Read: plugins/yzmir-dynamic-architectures/skills/using-dynamic-architectures/ml-lifecycle-orchestration.md
```
