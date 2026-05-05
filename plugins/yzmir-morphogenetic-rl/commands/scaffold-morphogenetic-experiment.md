---
description: Scaffold a new morphogenetic-RL experiment with controller, governor, replay log, and ablation-friendly telemetry
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<experiment-name> [--reward=utility_minus_cost|sparse|hybrid] [--seeds=10]"
---

# Scaffold Morphogenetic Experiment

Create a new morphogenetic-RL experiment with the structure required for the discipline this domain demands: separate RNG streams, replay logs, governor as a non-policy module, ablation-friendly telemetry, and the baselines required to prove the controller did anything at all.

## What Gets Created

```
experiment-name/
├── train.py                      # Host trainer + controller + governor wiring
├── controller.py                 # Policy proposing structural mutations
├── governor.py                   # Pre-flight, watch, rollback (non-policy)
├── morphogenesis.py              # Module insertion, alpha schedule, FSM ops
├── replay.py                     # Replay log writer + counterfactual replay
├── telemetry.py                  # Step/event/sidecar table writers
├── eval.py                       # Multi-seed evaluation harness
├── baselines/
│   ├── off_switch.py             # Same harness, controller disabled
│   ├── static_initial.py         # Static at initial shape
│   ├── static_final.py           # Static at morphogenetic final shape
│   └── fixed_schedule.py         # Hand-coded growth schedule, no learning
├── config.yaml                   # All hyperparameters separated
├── tests/
│   ├── test_determinism.py       # Asserts same seed → same event log
│   └── test_governor_invariants.py  # Asserts controller cannot disable gates
├── requirements.txt
└── README.md
```

This layout is non-negotiable in spirit. You can rename files; you cannot omit the categories. Each file represents a discipline this pack documents:

| File | Discipline |
|------|------------|
| `controller.py` | `rl-controller-for-morphogenesis.md` |
| `governor.py` | `governor-and-safety-gates.md` |
| `replay.py` | `deterministic-morphogenesis.md` |
| `telemetry.py` | `growth-telemetry-and-ablation.md` |
| `baselines/*` | `evaluation-under-topology-change.md`, `when-not-to-grow.md` |

If your scaffold lacks any of these, you are missing a discipline.

## Key Principles

### 1. Separate RNG Streams from Day One

```python
# train.py
@dataclass
class RNGStreams:
    trainer: torch.Generator
    controller: torch.Generator
    morphogenesis: torch.Generator
    governor: torch.Generator

def make_streams(master_seed: int, device: torch.device) -> RNGStreams:
    return RNGStreams(
        trainer=torch.Generator(device).manual_seed(master_seed),
        controller=torch.Generator(device).manual_seed(master_seed ^ 0xC011_4011),
        morphogenesis=torch.Generator(device).manual_seed(master_seed ^ 0x600B_0061),
        governor=torch.Generator(device).manual_seed(master_seed ^ 0x6045_0AAA),
    )
```

Anything else means your trainer's RNG state diverges the first time the controller fires, and your run is no longer reproducible. See `deterministic-morphogenesis.md`.

### 2. Governor Outside the Controller

```python
# governor.py — separate module, no controller import
@dataclass
class Governor:
    pre_event_window: deque
    watch_windows: dict[ActionId, PendingAction]
    cooldown: dict[SlotId, int]

    def pre_flight(self, state, action, step) -> Veto | Approval: ...
    def post_step(self, state, step) -> list[Rollback]: ...
```

The governor module must not import from `controller.py`. CI should fail on a circular import. This is the architectural enforcement of the "controller cannot disable governor" invariant. See `governor-and-safety-gates.md`.

### 3. Two Telemetry Tables, Constant Width

```python
# telemetry.py
class StepTable:
    columns = [
        "run_id", "step", "loss", "grad_norm", "grad_norm_clipped",
        "lr", "param_count", "flops_per_step",
        "active_module_count", "pending_event_count", "reward_mode",
    ]

class EventTable:
    columns = [
        "run_id", "event_id", "step", "kind", "slot_id", "action_type",
        "action_params", "controller_log_prob", "governor_reason",
        "pre_event_window_hash", "post_event_window_summary",
        "delta_param_count", "reward_mode",
    ]
```

Per-module statistics go in a sidecar table — never as columns in `StepTable`. See `growth-telemetry-and-ablation.md`.

### 4. Baselines Are Code, Not Aspirations

The `baselines/` directory must be runnable. Acceptance criterion for any morphogenetic claim is: all four baselines complete on the same harness, with the same seed range, before reporting numbers. See `evaluation-under-topology-change.md` and `when-not-to-grow.md`.

### 5. Determinism Is Tested in CI

```python
# tests/test_determinism.py
def test_same_seed_same_events():
    log_a = run_short_experiment(seed=42, steps=2000)
    log_b = run_short_experiment(seed=42, steps=2000)
    assert log_a.event_count == log_b.event_count
    for ev_a, ev_b in zip(log_a.events, log_b.events):
        assert ev_a == ev_b, f"divergence at event {ev_a.event_id}"
```

Run on every PR. The cost of finding non-determinism six months later is much higher than the cost of this test.

### 6. Governor-Invariant Tests

```python
# tests/test_governor_invariants.py
def test_governor_does_not_read_controller_output():
    governor = Governor(...)
    # The governor's interface must not accept controller-emitted fields
    # other than the ProposedAction itself. This is checked structurally.
    sig = inspect.signature(governor.pre_flight)
    assert "controller_confidence" not in sig.parameters
    assert "controller_recommendation" not in sig.parameters

def test_controller_cannot_modify_cooldown():
    # Static check that no path in controller.py writes governor.cooldown
    src = pathlib.Path("controller.py").read_text()
    assert "governor.cooldown" not in src
    assert "self.cooldown_override" not in src
```

These are soft tests. They make the anti-pattern visible in CI output rather than in the failure mode they prevent.

## Config Layout

```yaml
# config.yaml
master_seed: 42
device: cuda

trainer:
  batch_size: 64
  total_steps: 200000
  optimizer: adamw
  lr: 3.0e-4

controller:
  algorithm: ppo
  reward_mode: utility_minus_cost  # or sparse, hybrid
  action_space:
    slot_count: 16
    action_types: [grow, graft, retire, noop]
  rollouts_per_update: 16

governor:
  pre_event_window_size: 64
  watch_window_steps: 200
  cooldown_steps: 1000
  spike_k: 8.0
  sustained_k: 4.0

telemetry:
  step_table_path: logs/steps.parquet
  event_table_path: logs/events.parquet
  module_sidecar_path: logs/modules.parquet
  flush_every_steps: 1000

evaluation:
  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
  baselines: [off_switch, static_initial, static_final, fixed_schedule]
```

## Post-Scaffold Checklist

After scaffolding, before running the first real experiment:

1. **Run the determinism test**: `pytest tests/test_determinism.py -v`
2. **Run the governor-invariant tests**: `pytest tests/test_governor_invariants.py -v`
3. **Run a short morphogenetic run** (~2000 steps) and confirm step + event tables are populated
4. **Run the off-switch baseline** for the same length — confirm it differs from the morphogenetic run
5. **Run a counterfactual replay** — re-run with one event skipped, confirm divergence-from-skip-point is correct
6. **Check the schemas do not widen**: same number of columns in step/event tables before and after a grow event

If any of (1)-(6) fails, fix it now. The cost compounds.

## Anti-Patterns This Scaffold Prevents

| Anti-pattern | How the scaffold prevents it |
|--------------|------------------------------|
| Single shared seed | `make_streams` returns four streams |
| Controller disables governor | `governor.py` has no `import controller` |
| Schema widens on grow | Sidecar table keyed by `module_id`, not columns in step table |
| Off-switch baseline never run | `baselines/off_switch.py` exists; CI runs it on each release |
| No counterfactual replay | `replay.py` exposes a `run_counterfactual(skip_event_id=...)` API |
| Best-of-N reporting | `eval.py` requires `seeds` to be a list, refuses single-seed mode |
| No tests for invariants | `tests/` directory has both determinism and governor tests |

## Load Detailed Guidance

For controller / observation / reward design:
```
Load skill: yzmir-morphogenetic-rl:using-morphogenetic-rl
Then read: rl-controller-for-morphogenesis.md
```

For governor design:
```
Then read: governor-and-safety-gates.md
```

For determinism:
```
Then read: deterministic-morphogenesis.md
```

For telemetry schemas:
```
Then read: growth-telemetry-and-ablation.md
```

For baselines and fair comparison:
```
Then read: evaluation-under-topology-change.md
Then read: when-not-to-grow.md
```

## Reference

For the algorithmic substrate (PPO/SAC implementation, exploration):
```
Load skill: yzmir-deep-rl:using-deep-rl
```

For the host-side training mechanics (FSM, gradient isolation):
```
Load skill: yzmir-dynamic-architectures:using-dynamic-architectures
```
