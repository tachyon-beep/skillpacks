---
name: cost-estimation-and-compilation-budgets
description: "Use when estimating the runtime cost of a compiled tensor artifact, calibrating static estimates against measured cost, setting compile-time budgets, or reasoning about staleness \u2014 when compilation latency makes an artifact obsolete before it can be used."
---

# Cost Estimation and Compilation Budgets

## When to Use

- The compiler reports a cost estimate that downstream decisions depend on.
- Compilation is slow enough that people are working around it.
- Artifacts arrive after the decision that needed them.
- You are choosing between kernels or fusion strategies using predicted cost.

---

## Core Principle

**A cost estimate is a prediction, and an uncalibrated prediction is a number with the shape of evidence. Measure, compare, and record the error — or stop reporting the estimate, because a confidently wrong estimate is worse than no estimate.**

The second half of the sheet is a different failure: compilation is itself a cost, paid in latency, and an artifact that arrives after the moment it was needed has a value of zero regardless of quality.

### The failure this prevents

An uncalibrated cost model gets used for decisions. It ranks kernels, picks fusion strategies, and feeds admission decisions about whether an artifact is worth deploying. If it systematically underestimates memory-bound ops — the standard failure, because FLOP counts ignore bandwidth — then every decision it informs is biased the same way, and the bias is invisible because nothing compares prediction to reality.

The system does not fail loudly. It quietly makes slightly wrong choices forever, and the aggregate is a compiler that is measurably worse than a random kernel choice while reporting that it optimised.

---

## Static Cost Models

Start with the standard three-term model, and be explicit about what it ignores:

```python
def static_cost(node, shape_info, device_profile) -> dict:
    """FLOPs, bytes moved, and launch overhead — the three terms that matter.
    Returns components, never a single scalar: a scalar hides which term dominates,
    and which term dominates is the whole content of the estimate.
    """
    flops = flop_count(node, shape_info)
    bytes_moved = sum(numel(t) * dtype_bytes(t) for t in node_tensors(node))
    compute_s = flops / device_profile.peak_flops_per_s
    memory_s  = bytes_moved / device_profile.peak_bandwidth_bytes_per_s
    return {
        "flops": flops,
        "bytes": bytes_moved,
        "compute_bound_s": compute_s,
        "memory_bound_s": memory_s,
        "launch_overhead_s": device_profile.kernel_launch_overhead_s,
        # roofline: the slower of the two, plus overhead
        "estimate_s": max(compute_s, memory_s) + device_profile.kernel_launch_overhead_s,
        "regime": "compute" if compute_s > memory_s else "memory",
    }
```

`regime` earns its place. Fusion helps memory-bound chains (it removes round trips to memory) and does almost nothing for compute-bound ops. A cost model that returns a single scalar cannot tell you *why* an op is expensive, so it cannot tell you which optimisation would help — and a fusion pass driven by a scalar will happily fuse compute-bound ops for no gain while adding legality risk.

What the model ignores, and must be known to ignore: cache hierarchy, occupancy, tail effects on small shapes, and — for anything under roughly 10 µs — the fact that launch overhead dominates everything else. **On small shapes the estimate is essentially launch overhead × node count**, which is a genuinely useful prediction and not what most people expect a FLOP-based model to say.

---

## Calibration Is Not Optional

An estimate is only as good as its last comparison against measurement.

```python
import time, statistics, torch

def measure_cost(module, inputs, warmup=10, iters=50) -> float:
    """Median wall time per call. Median, not mean — one page fault ruins a mean."""
    for _ in range(warmup):
        module(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        module(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()          # without this you time the LAUNCH, not the work
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)

def calibrate(estimates: dict[str, float], measured: dict[str, float]) -> dict:
    """Report the error DISTRIBUTION and its direction, not a single accuracy number."""
    ratios = {k: measured[k] / estimates[k] for k in estimates if estimates[k] > 0}
    r = sorted(ratios.values())
    return {
        "median_ratio": statistics.median(r),                    # 1.0 = perfect
        "p90_ratio": r[int(0.9 * len(r))] if r else float("nan"),
        "systematic_bias": ("UNDERESTIMATES" if statistics.median(r) > 1.2
                            else "OVERESTIMATES" if statistics.median(r) < 0.8
                            else "calibrated"),
        "worst": sorted(ratios.items(), key=lambda kv: -abs(kv[1] - 1))[:5],
    }
```

Two things make this useful rather than decorative:

- **`torch.cuda.synchronize()`.** CUDA calls are asynchronous. Timing without a sync measures how long it took to *enqueue* the work — typically microseconds regardless of the kernel. Uncalibrated GPU cost models are frequently calibrated against nothing but launch latency, which is why they report suspiciously consistent numbers.
- **Systematic bias, not average error.** A model that is 30% off in both directions is noisy but unbiased and still ranks kernels correctly. A model that is 30% low *every time* on memory-bound ops mis-ranks every memory-bound-versus-compute-bound comparison it makes. Bias is the property that corrupts decisions; average error is not.

Run calibration in CI on a fixed op suite. When `systematic_bias` moves off `calibrated`, the model needs refitting — usually because a new device profile or a new op class was added without one.

---

## Compile-Time Budgets

Compilation costs time. Declare it up front, the same way you declare tolerances, and for the same reason: a budget agreed after the overrun is not a budget.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CompilationBudget:
    wall_clock_s: float
    peak_memory_bytes: int
    on_overrun: str    # "fail" | "degrade" | "abandon"

def compile_within_budget(spec, target, budget, manifest):
    for tier in ("full", "reduced", "minimal"):          # progressively fewer passes
        t0 = time.perf_counter()                          # per attempt — a clock that
        artifact = compile_artifact(spec, target, optimisation_tier=tier)
        spent = time.perf_counter() - t0                  # accumulated across tiers could
        manifest.append({"tier": tier, "spend_s": spent}) # never recover after one overrun
        if spent <= budget.wall_clock_s:
            artifact.compile_spend = {"wall_s": spent, "tier": tier}
            return artifact
        if budget.on_overrun == "fail":
            raise CompilationBudgetExceeded(f"{spent:.1f}s > {budget.wall_clock_s}s at "
                                            f"tier {tier!r}")
        if budget.on_overrun == "abandon":
            return None
        # "degrade": fall through and try a cheaper tier against a fresh clock
    raise CompilationBudgetExceeded("even the minimal tier exceeded the budget")
```

Degradation tiers matter because the alternative to a slow artifact is usually not a fast artifact — it is *no artifact*. A less-optimised artifact that arrives in time beats an optimal one that does not.

One honest limitation of overrun-then-degrade: the overrunning attempts still cost real wall time, so in `"degrade"` mode the budget bounds the *accepted artifact's* compile spend, not the total latency to obtain it. When total latency is the binding constraint, use the calibrated cost model above to pick the tier *before* compiling, rather than discovering it by overrun.

The critical constraint: **every tier produces an artifact that passes the same conformance gate.** Degradation may drop optimisations; it may never drop verification. A "fast path" that skips conformance is not a degraded compile, it is an unverified one, and the pressure to create it is highest exactly when the budget is tight.

Record the tier in the manifest. Two artifacts with the same semantic hash and different performance are explained by one field.

---

## Staleness: When Latency Makes an Artifact Obsolete

If the thing that requested compilation moves on while compilation runs, the artifact can be correct and useless. This is the failure mode of asynchronous compilation in any system where the surrounding state evolves — a training loop that has moved several thousand steps, a serving tier that has already rolled forward, a search that has abandoned the branch.

```python
@dataclass
class StalenessPolicy:
    max_age_s: float | None            # wall clock
    max_state_drift: float | None      # domain units: steps, epochs, config generations
    on_stale: str                      # "requalify" | "discard" | "use_with_warning"

def check_staleness(artifact, policy, now, current_state_position) -> tuple[str, str]:
    age = now - artifact.requested_at
    drift = current_state_position - artifact.requested_at_state_position
    if policy.max_age_s and age > policy.max_age_s:
        return policy.on_stale, f"age {age:.1f}s exceeds {policy.max_age_s}s"
    if policy.max_state_drift and drift > policy.max_state_drift:
        return policy.on_stale, (f"state moved {drift} units since the request; the "
                                 "artifact was compiled for a state that no longer exists")
    return "fresh", f"age {age:.1f}s, drift {drift}"
```

`requested_at` — not `compiled_at` — is the correct reference point. Staleness is measured from when the artifact was *asked for*, because that is when the requester's assumptions were true. Measuring from completion makes a slow compile look fresh the moment it finishes, which inverts the metric.

**Measure the staleness curve before choosing a policy.** Log request time, completion time, and state drift for a while, then set thresholds from the observed distribution. A guessed threshold either discards useful artifacts or admits obsolete ones, and you will not know which.

The design responses, in rough order of preference:

| Response | When |
|----------|------|
| Reduce compile latency (tiers, caching, fewer passes) | Always try first — it dominates the others |
| Cache aggressively on canonical identity | Repeated requests for the same semantics; see `artifact-identity-and-caching.md` |
| Compile speculatively, ahead of the request | The request set is predictable |
| Re-qualify on arrival | Cheap to re-check against current state |
| Fall back to an unoptimised path | Latency budget is hard and the artifact will not make it |

---

## Executable Decision Procedure: Is the Compilation Worth It?

```python
def compilation_worth_it(*, estimated_speedup: float, compile_time_s: float,
                         expected_uses: int, per_use_runtime_s: float,
                         cache_hit_probability: float, staleness_risk: float,
                         estimate_calibrated: bool) -> tuple[str, str]:
    if not estimate_calibrated:
        return "CALIBRATE-FIRST", ("The speedup estimate is uncalibrated. Measure a "
                                   "sample against prediction before deciding anything "
                                   "on it — an uncalibrated model biases every choice "
                                   "it informs in the same direction.")
    saved = expected_uses * per_use_runtime_s * (1 - 1 / max(estimated_speedup, 1e-9))
    effective_compile = compile_time_s * (1 - cache_hit_probability)
    expected_value = saved * (1 - staleness_risk) - effective_compile

    if expected_value <= 0:
        return "SKIP", (f"expected saving {saved:.2f}s vs effective compile cost "
                        f"{effective_compile:.2f}s (staleness risk {staleness_risk:.0%}) "
                        "— run unoptimised")
    if staleness_risk > 0.5:
        return "COMPILE-ASYNC-WITH-FALLBACK", (
            f"positive EV ({expected_value:.2f}s) but >50% chance of arriving stale. "
            "Start unoptimised, swap in the artifact if it lands fresh.")
    if effective_compile > saved * 0.5:
        return "COMPILE-REDUCED-TIER", (
            f"marginal: compile cost {effective_compile:.2f}s vs saving {saved:.2f}s. "
            "Use a reduced optimisation tier.")
    return "COMPILE", f"expected net saving {expected_value:.2f}s"
```

`expected_uses` is what most of these decisions actually turn on, and it is the input people skip. Compiling something used once is almost never worth it; compiling something used ten thousand times almost always is. `cache_hit_probability` folds in that the cost is amortised across everyone who will hit the same key.

---

## RED → GREEN Scenario

**RED.** A pipeline compiles candidate subgraphs asynchronously and admits them based on the compiler's reported runtime cost estimate. It runs for months. Analysis then shows admitted candidates are, on average, *slower* in production than the ones rejected.

Two compounding causes:

1. The cost model counts FLOPs and ignores memory bandwidth. Memory-bound candidates are systematically underestimated, so they look cheap and get admitted. Nobody noticed because nothing ever compared estimate to measurement.
2. Compilation takes 40–200 seconds. By the time an artifact lands, the state it was compiled against has moved on by thousands of steps. Some artifacts are stale on arrival and are used anyway, because staleness is unmeasured.

The system reports that it is optimising. It is anti-optimising, and every number it publishes is consistent with success.

**GREEN.**

1. **Calibrate.** Measure a sample of artifacts and compare against prediction. `systematic_bias` returns `UNDERESTIMATES` with a median ratio of 2.3× on memory-bound ops — the diagnosis, in one number.
2. **Split the model.** Report `compute_bound_s` and `memory_bound_s` separately and take the roofline max. Memory-bound candidates stop looking cheap.
3. **Measure the staleness curve** before setting a policy: log `requested_at`, `completed_at`, and state drift; the p50 is 40 s and the p95 is 200 s, against a tolerance of ~50 steps.
4. **Set the policy from the data**, with `on_stale="requalify"` — a stale artifact is re-checked against current state rather than trusted or discarded.
5. **Publish estimate-versus-measured in the manifest** for every artifact, so drift in the model surfaces as a trend rather than as a quarterly surprise.

Generalisable: **an unmeasured estimate is not a weak signal, it is an unknown-sign one.** The system had been making decisions with a number whose direction of error nobody had established, and confidence in it grew precisely because it was never checked. One calibration run answers the question.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Uncalibrated estimate used for decisions | Biased decisions, invisible | Calibrate in CI; report bias |
| Single scalar cost | Hides which term dominates; misdirects optimisation | Report compute/memory/launch separately |
| FLOPs only | Underestimates memory-bound ops systematically | Roofline: max(compute, memory) |
| Timing without `cuda.synchronize()` | Measures enqueue latency, not work | Synchronise before stopping the clock |
| Mean instead of median | One page fault dominates | Median with warmup |
| Reporting average error | Bias is what corrupts decisions | Report direction of bias |
| No compile-time budget | Artifacts arrive after they are needed | Declare a budget with tiers |
| Degradation tier that skips conformance | An unverified artifact, not a degraded one | Every tier passes the same gate |
| Staleness measured from completion | A slow compile looks fresh on arrival | Measure from `requested_at` |
| Guessed staleness threshold | Discards useful or admits obsolete | Measure the curve first |
| Compiling single-use artifacts | Compile cost exceeds any saving | Weigh `expected_uses` |
| Estimate-vs-measured not recorded | Model drift found quarterly, not weekly | Record both in the manifest |

---

## Checklist

- [ ] Cost model reports compute, memory, and launch terms separately
- [ ] Roofline (`max`) rather than sum of compute and memory
- [ ] `regime` exposed so fusion targets memory-bound chains
- [ ] Calibration runs in CI against measured cost
- [ ] Systematic bias reported, not just average error
- [ ] Measurement uses warmup, median, and `cuda.synchronize()`
- [ ] Compile-time budget declared with degradation tiers
- [ ] Every tier passes the same conformance gate
- [ ] Staleness measured from request time
- [ ] Staleness thresholds derived from an observed distribution
- [ ] `expected_uses` and cache-hit probability considered before compiling
- [ ] Estimate and measured cost both recorded in the manifest

---

## Related Sheets

- [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — cost as the ranking function after contract filtering
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — why the memory/compute regime drives fusion
- [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — caching as the primary latency fix
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — recording spend and estimates
- [conformance-testing.md](conformance-testing.md) — the gate no tier may skip
