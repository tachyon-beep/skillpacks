---
name: deterministic-morphogenesis
description: Use when reproducing morphogenesis runs across topology changes — same-seed-same-graft guarantees, RNG isolation between policy/governor/network, replay across shape changes, and the cross-pack interaction with axiom-determinism-and-replay.
---

# Deterministic Morphogenesis

## When to Use

- Standing up a new morphogenetic system and you want replay/ablation to actually work later
- Same seed and same data produce different topologies on rerun
- A failure happened once and cannot be reproduced for debugging
- Multi-rank training and ranks disagree on whether/where to grow
- Adding a replay log for offline analysis of controller decisions

For low-level numerical determinism in PyTorch (CUDA flags, deterministic algorithms), see `yzmir-simulation-foundations/check-determinism`. For the architecture-level treatment — determinism tiers, seed governance, replay infrastructure, and the cross-machine/floating-point questions this sheet deliberately does not re-derive — see `axiom-determinism-and-replay` (`seed-governance`, `rng-isolation-patterns`, `replay-infrastructure-design`, `floating-point-determinism`, `gpu-determinism`, `canonical-state-encoding-for-replay`, `divergence-detection-and-localisation`). This sheet covers the *additional* discipline morphogenesis requires on top of both.

---

## Core Principle

**Determinism in morphogenesis is load-bearing, not aspirational.**

If you cannot reproduce a topology from a seed, you cannot:

- Debug the controller (the failure is unreproducible)
- Ablate growth events (you cannot subtract one decision and rerun)
- Compare two policies (their runs diverge for non-policy reasons)
- Validate the governor (you cannot re-trigger a panic event)
- Trust any metric (run-to-run noise dwarfs the effect you are measuring)

You pay the cost upfront, or you pay it forever in unreproducible bug reports.

There are three sources of non-determinism specific to morphogenesis. Each must be controlled separately:

1. **Controller stochasticity** — the policy samples from a distribution
2. **Growth-event randomness** — initialization of the new module, slot selection ties, weight permutation
3. **Multi-rank disagreement** — different ranks compute different "should grow" verdicts

Framework-level non-determinism (CUDA reduction order, dropout, batch norm) is shared with the host trainer and is the subject of `yzmir-simulation-foundations/check-determinism`. This sheet assumes that level is solved.

---

## Determinism vs Reproducibility

These are different. Pick the one your system actually needs.

| Property | What it guarantees | Cost |
|----------|-------------------|------|
| **Deterministic given seed + data** | Same seed + same data + same code → same trajectory | Modest. Achievable on single rank with care. |
| **Bit-reproducible** | Same trajectory regardless of hardware, CUDA version, kernel choice | Expensive. Disables many optimizations. |
| **Replayable from log** | Trajectory can be reconstructed from a recorded decision log, even if RNG is non-deterministic | Cheap. Useful when full determinism is impractical. |

For most morphogenetic research substrates, **deterministic given seed + data** is the right target. Bit-reproducibility is needed for safety-critical work; if that is your tier, `axiom-determinism-and-replay/determinism-vs-reproducibility` and `floating-point-determinism` are where the cross-machine and FP-policy questions get answered — this sheet assumes same-hardware determinism and does not re-derive them. Replayability is a lighter alternative if you cannot achieve full determinism — log every controller decision and replay from the log.

A common failure: aiming for bit-reproducibility, achieving none of the three, and shipping a system that cannot be debugged.

---

## RNG Discipline for Growth Events

### The Rule

**The controller, the growth-event sampler, and the host trainer use separate RNG streams.**

A single shared `torch.manual_seed(42)` will not give you what you want. The first time the controller decides to grow, the host trainer's RNG state diverges from any rerun where the controller decided not to grow at the same step. Every subsequent batch shuffle, dropout mask, and stochastic op silently changes.

```python
# WRONG: one stream, divergence on first morphogenetic decision
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
# controller, init, dropout all share state
# any change in controller decisions invalidates the rest of the trajectory
```

```python
# RIGHT: separate streams, isolated by purpose
@dataclass
class RNGStreams:
    trainer: torch.Generator       # batch shuffle, dropout, augmentation
    controller: torch.Generator    # policy action sampling
    morphogenesis: torch.Generator # init of new modules, slot tiebreaks
    governor: torch.Generator      # if any (rare; governors are usually deterministic)

def make_streams(master_seed: int, device: torch.device) -> RNGStreams:
    g_trainer = torch.Generator(device=device).manual_seed(master_seed)
    g_controller = torch.Generator(device=device).manual_seed(master_seed ^ 0xC011_4011)
    g_morph = torch.Generator(device=device).manual_seed(master_seed ^ 0x600B_0061)
    g_gov = torch.Generator(device=device).manual_seed(master_seed ^ 0x6045_0AAA)
    return RNGStreams(g_trainer, g_controller, g_morph, g_gov)
```

The trainer's RNG state at any step is now a function of `(master_seed, step)` only — independent of how many times the controller fired or what it decided.

### What Each Stream Owns

| Stream | Owns |
|--------|------|
| Trainer | Dataloader shuffle, dropout, batch norm running stats updates that need RNG, augmentation |
| Controller | Action sampling from the policy distribution, exploration noise |
| Morphogenesis | New-module weight init, slot-tiebreak randomness, blend-schedule jitter |
| Governor | Tiebreaks on coincident panic events (very rare; usually deterministic) |

If a single op needs randomness from multiple streams, that is a code smell. Pick one. Document why.

**Ownership is not always enforceable by passing a generator.** `torch.nn.Dropout` and `F.dropout` take no `generator=` argument (nor do most fused CUDA kernels), so the trainer stream cannot *own* dropout the way it owns dataloader shuffling. For ops with no `generator=` parameter, isolate by scoping the global state instead:

```python
with torch.random.fork_rng(devices=[device]):
    torch.manual_seed(trainer_epoch_seed)   # derived from the trainer stream
    loss = model(batch)                     # dropout draws from the forked global state
# global RNG state restored on exit; the controller's stream was never touched
```

Record every such op as an explicit exception in your stream table. Anything you can neither pass a generator to nor fork around is an un-isolated stream — log it as a known determinism gap rather than assuming the table covers it.

**Draw on the generator's device.** A CUDA generator cannot seed a CPU tensor factory (`torch.randint(..., generator=cuda_gen)` raises `Expected a 'cpu' device type for generator but found 'cuda'`), and `make_streams` above builds device-resident generators. Pass `device=rng.device` at every draw site, or build the control-plane streams on CPU deliberately and document that choice.

### Per-Event Sub-Streams

For replay surgery — "rerun, but skip event #17" — you need to derive a fresh sub-stream for each growth event so its randomness is independent of event ordering:

```python
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF
_GOLDEN = 0x9E37_79B9_7F4A_7C15   # odd 64-bit constant; the multiply is a bijection mod 2^64

def event_rng(parent: torch.Generator, event_id: int) -> torch.Generator:
    """Deterministic per-event RNG. Independent of how many events preceded."""
    # The mask is load-bearing, not cosmetic: `event_id * _GOLDEN` exceeds 2**64
    # from event_id = 2 onward, and manual_seed then raises
    # "Overflow when unpacking long long" — i.e. the run dies on its second
    # growth event. Reduce mod 2**64 before seeding.
    seed = (parent.initial_seed() ^ (event_id * _GOLDEN)) & _MASK64
    return torch.Generator(device=parent.device).manual_seed(seed)
```

Now removing event #17 from the replay does not perturb event #18's randomness.

The returned generator inherits the parent's device, so draws from it must name that device too — `torch.rand(n, generator=g, device=g.device)`.

---

## Replay Logs

A **replay log** records every controller decision and every governor verdict, with enough state for offline reconstruction.

### What to Log per Decision

```python
@dataclass(frozen=True)
class ControllerDecision:
    step: int
    event_id: int                    # monotonic, used as RNG salt
    observation_hash: str            # sha256 of the observation tensor
    action: ProposedAction           # the action object as proposed
    log_prob: float                  # for off-policy correction during replay
    sampled_seed: int                # the per-event seed actually used
    policy_version: str              # checkpoint id of the controller weights

@dataclass(frozen=True)
class GovernorVerdict:
    step: int
    event_id: int
    decision: Literal["approve", "veto", "rollback"]
    reason: str                      # structured: "non_finite_loss", "cooldown", "loss_spike"
    pre_event_window_hash: str       # sha256 of frozen pre-event window
    panic_rule_fired: str | None     # which rule, if any
```

### Why Log the Hashes

You will eventually re-run a replay against modified code and want to know whether the divergence point is in your code or in your RNG. Hashes of observations and pre-event windows let you locate the exact step where the new code starts producing different state.

### Replay Modes

| Mode | What it does | Use case |
|------|--------------|----------|
| **Full replay** | Re-run from scratch with the same seed; ignore the log | Verify your determinism claim |
| **Decision replay** | Re-run; at each event, force the logged action; verify governor verdict matches | Test new governor logic against historical decisions |
| **Counterfactual replay** | Re-run; force a *different* action at one event; observe divergence from event onward | Ablation: "what if the controller had not grown at step 4000?" |
| **Partial replay** | Reconstruct from a checkpoint plus the log tail | Long runs; debug without restarting |

A system that supports all four is debuggable. A system that supports none is a one-shot experiment.

---

## Multi-Rank Synchronization

If training is distributed, the controller's decision must be agreed across ranks before any mutation happens. Otherwise different ranks build different networks and gradient sync explodes.

### The Pattern

**One rank decides. Decisions broadcast. All ranks apply the same mutation.**

```python
def step_controller_distributed(controller, observation, world_size, rank):
    if rank == 0:
        action = controller.act(observation)  # only rank 0 samples
    else:
        action = None

    # Broadcast the decision (serialize ProposedAction first; see below).
    # broadcast_object_list mutates its list argument IN PLACE and returns None.
    # Read the result back out of the buffer you passed in — subscripting the
    # return value raises TypeError, and passing a throwaway list literal drops
    # the broadcast result on the floor, leaving every non-src rank with None.
    buf = [action]
    dist.broadcast_object_list(buf, src=0)
    action = buf[0]

    return action  # all ranks now have the same ProposedAction
```

Two subtleties:

**Subtlety 1**: The controller's *observation* must be the same on rank 0 every run, or the policy decides differently each time. If your observation includes any rank-0-local statistic (a single GPU's loss, say), make it a global all-reduced statistic before passing it to the controller.

**Subtlety 2**: If the controller is itself distributed (e.g., for very large policies), you need consensus among controller ranks too. In practice, almost no morphogenetic controller is large enough to need sharding. Keep the controller small and run it on rank 0 only.

### Governor on Distributed Training

The governor's panic-detection inputs (loss, grad norm) are global statistics. All-reduce them before feeding the governor. Do not let rank 0's loss-spike decision drift from rank 7's.

```python
# all_reduce is IN PLACE and returns None (assigning its result gives you None).
# Two further traps: reducing `loss_local` directly mutates a tensor autograd is
# still holding, and ReduceOp.AVG is NCCL-only (Gloo rejects it). Clone off the
# graph, reduce with SUM, divide by world_size — portable and autograd-safe.
loss_global = loss_local.detach().clone()
dist.all_reduce(loss_global, op=dist.ReduceOp.SUM)
loss_global /= world_size

grad_norm_global = grad_norm_local.detach().clone()
dist.all_reduce(grad_norm_global, op=dist.ReduceOp.SUM)
grad_norm_global /= world_size

verdict = governor.post_step(state.with_global_stats(loss_global, grad_norm_global), step)
```

The governor must see the *same* scalars on every rank. Reducing with SUM ÷ `world_size` rather than AVG also keeps the same code path working on a Gloo CPU test rig, which is where you will actually run your determinism CI.

### Action Serialization Across Ranks

`ProposedAction` must round-trip through `dist.broadcast_object_list` deterministically. Use plain dataclasses with primitive fields. Avoid:

- Tensors as action fields (broadcast separately)
- `torch.Generator` references in actions (they don't pickle deterministically)
- Floats that came from a non-deterministic op (round to a fixed precision)

---

## Validating Determinism

A determinism claim is empty without a test. Add this to CI or to your experiment-launcher:

```python
def assert_morphogenesis_deterministic(experiment_fn, master_seed: int, steps: int) -> None:
    log_a = experiment_fn(master_seed=master_seed, steps=steps)
    log_b = experiment_fn(master_seed=master_seed, steps=steps)

    if log_a.event_count != log_b.event_count:
        raise NonDeterminismError(
            f"Event count differs: {log_a.event_count} vs {log_b.event_count}"
        )

    for ev_a, ev_b in zip(log_a.events, log_b.events):
        if ev_a != ev_b:
            raise NonDeterminismError(
                f"First divergence at event {ev_a.event_id}, step {ev_a.step}: "
                f"a={ev_a}, b={ev_b}"
            )
```

Run this every time you change anything controller-, governor-, or morphogenesis-adjacent. The cost of finding a non-determinism bug six months later is far higher than the cost of running this test on every PR.

### Locating Divergence

When the test fails, the first divergence point is the bug. Walk back through the log:

1. Same observation on both runs at the divergence step? If no, the trainer's RNG drifted upstream — your stream-isolation is broken.
2. Same observation, different sampled action? The controller's RNG stream is broken.
3. Same action, different governor verdict? The governor reads non-deterministic state. Find what.
4. Same verdict, different post-event evolution? Framework-level non-determinism — see `yzmir-simulation-foundations/check-determinism`.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Single shared RNG | Trainer trajectory diverges on first controller change | Separate streams per purpose |
| `torch.manual_seed` only at start | Lazy CUDA init non-determinism | Use `torch.Generator` instances; pass them explicitly to every op that accepts `generator=`, and `torch.random.fork_rng` around those that do not (dropout) |
| Logging decisions but not seeds | Replay produces different per-event randomness | Log `sampled_seed` per event |
| Per-event seed = `master_seed + event_id` | Linear addition collides easily; predictable | Use a wide multiply or hash mix — masked, per the next row |
| Wide-multiply seed mix left unmasked | `event_id * 0x9E37…` exceeds 2**64 at `event_id = 2`; `manual_seed` raises "Overflow when unpacking long long" and the run dies on its second growth event | `(master_seed ^ event_id * 0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF_FFFF` |
| Draw from a device generator into a CPU tensor factory | `Expected a 'cpu' device type for generator but found 'cuda'` | Pass `device=rng.device` at the draw site |
| Assigning the result of `all_reduce` / `broadcast_object_list` | Both are in-place and return `None`; the variable silently becomes `None` | Reduce/broadcast into a buffer, then read the buffer |
| All-reduce after the controller decided | Rank 0 sampled on stale local state | All-reduce observations *before* feeding the controller |
| Governor reads `time.time()` for cooldowns | Wall-clock ≠ deterministic | Use `step` count |
| `random.shuffle` on a Python list as a slot tiebreak | Uses Python's RNG, not your stream | Pass an explicit `random.Random` instance |
| Action fields contain CUDA tensors | Broadcast object list non-deterministic | Move to CPU; round if floats |

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "Determinism is a nice-to-have; we'll add it later" | You will not. Adding it later means rewriting the controller, governor, and host trainer simultaneously. |
| "Setting `torch.manual_seed` is enough" | It controls one stream. Morphogenesis has at least three. |
| "The controller's decisions are noisy anyway, exact reproduction is unrealistic" | "Noisy when you want it" and "non-deterministic always" are different. The noise should be *reproducible* noise. |
| "Distributed training is inherently non-deterministic" | It is harder, not impossible. Single-rank determinism is the floor; distributed determinism is engineering work, not a wall. |
| "Bit-reproducibility is too expensive" | Probably true. Aim for deterministic-given-seed instead. They are different goals. |
| "We log losses; we can debug from those" | Losses are an aggregate. Debugging the controller needs the action sequence, which you do not have. |
| "We can re-derive the topology from final weights" | You cannot. Two different decision sequences can produce identical final shapes by coincidence and very different ones by design. |

---

## Red Flags Checklist

- [ ] **Single shared seed across trainer/controller/morphogenesis**
- [ ] **No replay log** — only loss curves are recorded
- [ ] **Event seeds are `master + event_id`** (linear, collision-prone)
- [ ] **Distributed training without rank-0 broadcast** — ranks decide independently
- [ ] **Governor reads wall-clock time** for cooldowns
- [ ] **No determinism CI test** — claim is unverified
- [ ] **Action objects contain non-picklable fields** (tensors, generators)
- [ ] **Per-event randomness depends on event order** — removing one event perturbs all subsequent ones
- [ ] **All-reduce happens after the controller acts** instead of before
- [ ] **Replay log lacks observation hashes** — divergence point cannot be located

---

## Diagnostic Questions

1. **What seeds does your system have?** If the answer is "one," you have a problem.
2. **What does your replay log record per event?** If it does not include the sampled seed and observation hash, replay is partial at best.
3. **What is your distributed broadcast pattern?** If ranks decide independently, you are in undefined territory.
4. **When did you last run the determinism test?** If the answer is never, run it now.
5. **What happens if you remove one event from the log and replay?** If the answer is "everything after diverges due to RNG," your per-event RNG depends on event order — fix the salt.
6. **What does the governor read?** If anything is non-deterministic in that path, your panic decisions are non-replayable.

---

## Cross-References

- **Low-level training determinism (CUDA, autograd, dropout)**: `yzmir-simulation-foundations/check-determinism`
- **Architecture-level determinism** — tier selection, seed governance, RNG isolation as a general pattern, replay infrastructure, snapshotting, divergence localisation: `axiom-determinism-and-replay` (`determinism-vs-reproducibility`, `seed-governance`, `rng-isolation-patterns`, `replay-infrastructure-design`, `snapshot-strategy`, `divergence-detection-and-localisation`)
- **Cross-machine and floating-point determinism** (the questions this sheet declares out of scope): `axiom-determinism-and-replay/floating-point-determinism`, `axiom-determinism-and-replay/gpu-determinism`
- **Canonical encoding of replay state** (when morphogenesis output must be byte-comparable across runs): `axiom-determinism-and-replay/canonical-state-encoding-for-replay`
- **Controller action / observation design** (which feeds the RNG-discipline boundary): `rl-controller-for-morphogenesis.md`
- **Governor's role in the deterministic pipeline**: `governor-and-safety-gates.md`
- **Logging schemas that survive topology change**: `growth-telemetry-and-ablation.md`
- **Fair comparison of replays with different controllers**: `evaluation-under-topology-change.md`
- **Multi-seed credit assignment under deterministic replay**: `multi-seed-coordination-rl.md`
