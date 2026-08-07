---
name: effect-sizes-and-cost-charged-utility
description: Use when deciding what quantity to test, when a raw metric improvement ignores what it cost, or when a statistically significant result is too small to act on. Covers cost-charged utility with a real zero, admission versus retention weights, practical versus statistical significance, and why standardized effect sizes are not comparable across designs.
---

# Effect Sizes and Cost-Charged Utility

## Overview

**Test the quantity you would act on. A raw metric improvement that ignores the compute it burned, the parameters it added, and the disruption it caused is not the thing you are deciding about — and optimising it will reliably select the most expensive candidate.**

This sheet defines the utility the rest of the pack tests, thresholds, powers, and reports. It has one non-negotiable property: **doing nothing scores exactly zero**, which is what makes "no candidate was worth it" a representable answer.

## When to Use

Use this sheet when:

- Choosing the dependent variable for a counterfactual trial.
- A candidate wins on the headline metric but costs substantially more.
- A result is statistically significant and someone asks "but is it worth it?"
- You need a `δ` for a power calculation and don't know where it comes from.
- The same candidate is evaluated at admission time and again for retention, and the two decisions disagree.

Do not use this sheet for:

- The statistical test applied to the utility — [paired-comparison-methods.md](paired-comparison-methods.md).
- Thresholding the utility for an act/abstain decision — [abstention-and-calibration.md](abstention-and-calibration.md).
- Reporting the quality/cost trade-off surface — [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md).

## Core Principle

> Charge every cost the decision actually incurs, in the same units as the benefit, with weights fixed before the data arrives. Then "is it significant?" and "is it worth it?" become the same question — asked of one number, against a zero that means something.

## The Utility Form

For candidate `c`, starting state `s`, horizon `H`:

```
u_admit(c, s, H) = M_control(s,H) − M_c(s,H)   −  λ·C_c  −  μ·S_c  −  ν·P_c
                   \__________________________/     \___/    \___/    \___/
                     raw benefit vs the no-op       compute  one-off  persistent
                     branch (positive = better)     charge   shock    resource
```

- `M` is the primary metric (loss, error, regret — anything where lower is better; flip signs otherwise).
- `C_c` — **consumable** cost incurred to produce and evaluate the candidate: host-equivalent forward/backward passes, generation, compilation, screening, wall clock.
- `S_c` — **one-off transition** cost: integration shock, the temporary degradation from installing a change, migration risk, review effort.
- `P_c` — **persistent** cost the system carries afterwards: added parameters, memory, latency, maintenance surface.
- `λ, μ, ν` convert each cost into metric units. They are the honest statement of your exchange rate, and they must be declared before results exist ([preregistration…](preregistration-and-exploratory-vs-confirmatory.md)).

The control branch scores `u = 0` identically: it has no benefit and no cost. That is what makes the zero real rather than nominal, and it is what [abstention](abstention-and-calibration.md) thresholds against.

## Why It Changes Answers

Three candidates, weights `λ = 0.8`, `μ = 1.5`, `ν = 0.0004` per 1k parameters:

| Candidate | raw improvement | `λC` | `μS` | `νP` | **`u_admit`** | `u_retain` | break-even raw |
|---|---|---|---|---|---|---|---|
| A (small) | 0.0210 | 0.0048 | 0.0030 | 0.0016 | **+0.0116** | +0.0146 | 0.0094 |
| B (large) | **0.0380** | 0.0144 | 0.0083 | 0.0124 | **+0.0029** | +0.0112 | 0.0350 |
| C (cheap) | 0.0100 | 0.0009 | 0.0006 | 0.0003 | **+0.0082** | +0.0088 | 0.0018 |

On the raw metric, B wins by a wide margin — it is nearly twice A. On cost-charged utility, **A wins and B is barely distinguishable from doing nothing.** B has to deliver 0.035 of raw improvement just to break even; it delivered 0.038.

The fragility is the point. If compute becomes scarce and `λ` doubles:

```
  A:  +0.0068     C:  +0.0073     B:  -0.0115   <- now actively harmful
```

A selection rule on the raw metric would have admitted B, and B is the candidate whose value evaporates under a plausible shift in operating conditions. **A system that selects on the uncharged metric systematically accumulates its most expensive options** — each individually justified by a headline number, collectively a cost problem nobody decided to take on.

## Admission vs Retention: same weights, different terms

The same candidate is evaluated twice in its life: *should we install it?* and later *should we keep it?* These are genuinely different questions, and the difference is principled:

```
u_admit(c,s,H)   = benefit − λC − μS − νP      # includes the one-off transition cost
u_retain(c,s,H)  = benefit − λC      − νP      # already installed; S is sunk
```

Dropping `S` for retention is correct — the shock is spent. In the table, B's retention utility (+0.0112) is nearly four times its admission utility (+0.0029): it is a candidate that is not worth installing but is worth keeping once installed. That asymmetry is real and useful, and a system that uses one utility for both decisions will get one of them wrong.

But the *shared* weights must genuinely be shared. **Using `λ = 0.8` at admission and `λ = 0.3` at retention silently guarantees that everything admitted is also retained**, which makes retention review theatre. If a weight legitimately differs between the two decisions, document the structural reason in the pre-registration; an undocumented difference is the tell that weights were tuned to produce an outcome.

The same discipline applies across candidates: one candidate charged for its integration shock while another is not makes the comparison meaningless, and the uncharged one always wins.

## Where the Weights Come From

Weights are a modelling choice, not a discovered fact. Three defensible derivations:

- **Budget-constrained** — if the system has a fixed compute budget, `λ` is the shadow price: the metric improvement obtainable by spending one unit of compute on the *next best alternative* (more training, more data, a bigger model). A candidate that improves the metric less per FLOP than plain extra training has negative utility, and should.
- **Willingness-to-pay** — ask the decision owner: "how much metric improvement would justify permanently adding 10k parameters?" Their answer is `ν`, and writing it down converts a recurring argument into a fixed constant.
- **Revealed from past decisions** — fit weights that rationalise decisions the organisation already made and endorses. Useful for calibration; be wary of encoding past mistakes.

Whatever the derivation, **run a sensitivity analysis and report it**: does the ranking survive a 2× change in each weight, one at a time? A conclusion that holds only at exactly `λ = 0.8` is a conclusion about `λ`, not about the candidate. The table above fails this test for B and passes it for A and C — which is far more informative than either point estimate.

## Practical vs Statistical Significance

Four cases, and only one of them is a decision:

| | Statistically significant | Not significant |
|---|---|---|
| **Practically meaningful** (CI excludes the decision threshold from above) | **Act.** | Underpowered — report the MDE and either extend or say inconclusive ([power](power-and-sample-size-for-paired-designs.md)) |
| **Practically negligible** (whole CI below the threshold) | **Do not act.** A real but worthless effect. Say so plainly. | **Do not act**, and stop testing this. |

Set a **decision threshold** `δ_min` — the smallest utility worth acting on — in the pre-registration, and report the interval against it rather than against zero. `δ_min` is also the `δ` for the power calculation, which is why sizing a fleet from a pilot's point estimate is the wrong move: `δ_min` comes from the cost model, not from the data.

The top-right cell deserves emphasis because it is the one people fight about: a large fleet will eventually make a trivial effect significant. "Significant at n = 400, and the whole confidence interval sits below the threshold that would justify the change" is a clean, defensible *no*.

**Standardised effect sizes do not transfer between designs.** `d_z = δ / sd_d` is useful within one experiment and misleading across them, because `sd_d` depends on how well the branches were matched. The same intervention scores `d_z = 0.16` without CRN and `d_z = 0.40` with it ([common-random-numbers…](common-random-numbers-and-matching.md)) — a 2.5× difference in "effect size" from a harness change. Report raw utility in interpretable units and state `sd_d` separately; quote `d_z` only alongside the design that produced it.

## Runnable: cost-charged utility with sensitivity

```python
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class CostWeights:
    """FROZEN before the trial. Record in preregistration.yaml with provenance."""
    lam: float     # per unit of consumable compute
    mu: float      # per unit of one-off transition/integration shock
    nu: float      # per unit of persistent resource (e.g. per 1k parameters)

def utility(metric_control, metric_candidate, compute, shock, persistent, w: CostWeights,
            phase="admission"):
    """Cost-charged utility. The no-op scores EXACTLY 0 (no benefit, no cost).

    phase='admission' charges the one-off shock; phase='retention' does not,
    because it is already sunk. Weights are otherwise IDENTICAL across phases --
    differing weights make retention review unfalsifiable.
    """
    benefit = metric_control - metric_candidate            # positive = better
    cost = w.lam * compute + w.nu * persistent
    if phase == "admission":
        cost += w.mu * shock
    elif phase != "retention":
        raise ValueError(f"unknown phase {phase!r}")
    return benefit - cost

def break_even(compute, shock, persistent, w: CostWeights, phase="admission"):
    """Raw improvement this candidate must deliver just to be worth zero.
    Quote it next to every candidate -- it makes the cost legible."""
    c = w.lam * compute + w.nu * persistent
    return c + (w.mu * shock if phase == "admission" else 0.0)

def weight_sensitivity(candidates, w: CostWeights, factors=(0.5, 2.0)):
    """Does the RANKING survive plausible weight changes? A conclusion that holds
    only at the nominal weights is a conclusion about the weights.

    candidates: [{"name": str, "metric_control":.., "metric_candidate":..,
                  "compute":.., "shock":.., "persistent":..}, ...]
    """
    def rank(w_):
        scored = [(c["name"], utility(**{k: v for k, v in c.items() if k != "name"}, w=w_))
                  for c in candidates]
        return [n for n, _ in sorted(scored, key=lambda kv: -kv[1])]

    out = {"nominal": rank(w)}
    for field in ("lam", "mu", "nu"):
        for f in factors:
            w2 = CostWeights(**{**w.__dict__, field: getattr(w, field) * f})
            out[f"{field}x{f}"] = rank(w2)
    out["ranking_is_robust"] = all(v == out["nominal"] for k, v in out.items() if k != "nominal")
    return out

W = CostWeights(lam=0.8, mu=1.5, nu=0.0004)
print(utility(2.400, 2.379, compute=0.0060, shock=0.0020, persistent=4.0, w=W))   # A: +0.0116
print(utility(2.400, 2.362, compute=0.0180, shock=0.0055, persistent=31.0, w=W))  # B: +0.0029
print(break_even(0.0180, 0.0055, 31.0, W))                                        # B needs 0.0350
```

## Decision Procedure

```
1. Name the primary metric M and its direction. One metric
   (multiple-comparisons-and-sequential-testing.md wants one primary endpoint).

2. Enumerate the costs the DECISION actually incurs, in three buckets:
   consumable (C), one-off transition (S), persistent (P). If a cost does
   not change with the decision, leave it out -- charging fixed costs adds
   noise without changing any ranking.

3. Set lambda / mu / nu by budget shadow price, willingness-to-pay, or
   revealed preference. Write the derivation down, not just the number.

4. Freeze the weights in preregistration.yaml. Use the SAME weights at
   admission and retention; the only permitted difference is dropping the
   sunk one-off shock term at retention.

5. Verify the zero: the no-op branch must score exactly 0. If it doesn't,
   the anchor is broken and abstention is undefined.

6. Set delta_min = the smallest utility worth acting on. This is the delta
   for power (power-and-sample-size-for-paired-designs.md) and the reference
   point for reporting intervals -- not zero.

7. Compute per-candidate break-even raw improvement and publish it next to
   the utility. It is the most legible way to show what a candidate costs.

8. Run the weight sensitivity (0.5x / 2x on each weight). If the ranking
   flips, report that -- it is a finding about the operating regime, not a
   caveat to bury.

9. Report utility in raw interpretable units. Quote d_z only with the
   design that produced it; it does not transfer across harnesses.
```

## RED Scenario

> A leaderboard ranks candidates by validation-loss improvement. The top entry improves loss by 0.038 — nearly double the runner-up's 0.021 — and is admitted automatically. Over two quarters the system's parameter count grows 40% and inference latency doubles, with aggregate metric gains that nobody can attribute.

**The catch:** the leaderboard optimises the uncharged metric, so it systematically selects candidates that buy improvement with resources. With the declared weights, the 0.038 candidate has a break-even of 0.0350 — it clears it by 0.0029, roughly a quarter of what the "worse" 0.021 candidate delivers (+0.0116). Two quarters of selecting on raw improvement is exactly how a parameter budget disappears without a decision ever being made to spend it.

**GREEN behaviour:** *"The ranking metric is the bug, not the candidates. Switch the leaderboard to cost-charged utility: `u = benefit − λC − μS − νP`, with λ, μ, ν declared up front and frozen. On the current weights that reorders the top two — the 0.038 candidate nets +0.0029 while the 0.021 candidate nets +0.0116 — and it explains the parameter growth: the system has been buying 0.038-sized improvements at 0.035-sized prices, all quarter.*
>
> *Three things to add alongside: (1) publish each candidate's break-even raw improvement — for the 0.038 candidate that is 0.0350, which makes the margin visible at a glance; (2) run the weight sensitivity, because at 2× λ that candidate goes to −0.0115 and its admission was regime-dependent, not robust; (3) apply the same weights at retention, dropping only the sunk integration term — otherwise everything admitted is automatically retained and the retention gate does nothing. Finally, set `δ_min` from the cost model and report intervals against it: with a big enough fleet, a +0.0029 utility will eventually be statistically significant, and 'significant but below the threshold that would justify the change' is the correct decision, not a disappointment."*

## Cross-References

- [abstention-and-calibration.md](abstention-and-calibration.md) — thresholding this utility; the zero it needs
- [paired-comparison-methods.md](paired-comparison-methods.md) — testing the per-unit utility difference
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — `δ_min` from the cost model becomes the power target
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — when to report the frontier instead of collapsing to one utility
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — freezing weights and their provenance
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — selecting on utility is still selecting
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-15 (uncharged metric), AP-16 (inconsistent weights across decisions)
