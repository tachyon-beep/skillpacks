---
name: power-and-sample-size-for-paired-designs
description: "Use when asked how many runs, seeds, or trajectories an experiment needs, when planning a fleet from pilot variance, or when a null result needs to be distinguished from an underpowered one. Covers power from the paired-difference SD at the unit level, minimum detectable effect, pilot-variance uncertainty, and why underpowered fleets exaggerate the effects they do find."
---

# Power and Sample Size for Paired Designs

## Overview

**Power is denominated in *units*, and it is driven by the standard deviation of the paired difference *at the unit level* — not by the spread of the outcome, and not by the number of branches. Get that one quantity right and the arithmetic is trivial; get it wrong and the fleet size is off by an order of magnitude.**

The two questions this sheet answers before you spend GPU time: *how many independent runs does this claim need?* and *given the fleet I can afford, what is the smallest effect I could detect?* The second question is the more useful one, because it is usually the honest answer to "can we just run 12?"

## When to Use

Use this sheet when:

- Someone asks "how many seeds/runs/trajectories do we need?"
- You have pilot data and need to size the confirmatory fleet.
- A trial returned a null and you need to say whether it was a real null or an underpowered one.
- Budget forces a fixed fleet size and you need to state what claim it can support.
- A multiple-comparison correction is planned and you need to know what it costs in `n`.

Do not use this sheet for:

- Deciding what a unit is — [statistical-units-and-clustering.md](statistical-units-and-clustering.md). **Power computed on the wrong unit is the most expensive arithmetic error in this pack.**
- Reducing the variance in the first place — [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md), which is usually a much better investment than more runs.

## Core Principle

> The input to a paired power calculation is `sd_d`: the standard deviation, across independent units, of the one-number-per-unit paired difference. Not the SD of the outcome. Not the SD across branches. If you can only estimate one thing from a pilot, estimate that.

The standardised effect is `d_z = δ / sd_d`, and for a two-sided paired *t* test at level α with power `1−β`:

```
n  ≈  ( (z_{1-α/2} + z_{1-β}) / d_z )²          # normal approximation, then round up
```

Refine with the noncentral *t* (adds 1–2 units at these sizes). The normal form is fine for planning; use the exact form for the pre-registration.

## Variance Reduction Beats Sample Size

The same `δ = 0.012` at two different `sd_d` values — the two configurations from [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md):

| Configuration | `sd_d` | `d_z` | runs for 80% power | power at 24 runs |
|---|---|---|---|---|
| Independent streams (no CRN) | 0.075 | 0.16 | **309** | 11% |
| Matched branches (CRN) | 0.030 | 0.40 | **52** | 47% |

**Fixing the harness saves 257 training runs.** Power scales with `1/sd_d²`, so halving the difference SD quarters the fleet. Before you ask for more compute, ask what is still unmatched between the branches — it is almost always the cheaper fix, and unlike more runs, it improves every future experiment too.

## Power at Realistic Fleet Sizes

`δ = 0.012`, `sd_d = 0.030`, α = 0.05 two-sided:

```
 runs   power     |   runs   MDE at 80% power
    8    0.16     |     12   0.0267
   12    0.24     |     24   0.0179
   16    0.32     |     52   0.0119
   24    0.47     |
   32    0.59     |
   52    0.81     |
   64    0.88     |
```

The MDE column is the one to quote when the fleet size is fixed by budget. "We can afford 24 runs" translates to "we can detect effects of 0.018 or larger" — which is a statement a stakeholder can evaluate. If the effect you care about is 0.012, 24 runs cannot resolve it and the experiment should be re-scoped, not run and hoped over.

## The Cost of Underpowered Fleets

An underpowered experiment is not merely *likely to miss* a real effect. When it does find one, the finding is **wrong in magnitude**, because only the luckiest samples clear the significance bar. Simulating true `δ = 0.012`, `sd_d = 0.030`:

| runs | power | mean \|effect\| among *significant* results | exaggeration | sign errors |
|---|---|---|---|---|
| 12 | 0.24 | 0.0221 | **1.8×** | 0.23% |
| 24 | 0.47 | 0.0170 | **1.4×** | 0.02% |
| 52 | 0.81 | 0.0134 | 1.1× | 0.00% |

This is the **Type-M (magnitude) error**, and it is why a literature of small studies reports effects that shrink on replication. A 24-run fleet that finds significance will, on average, report an effect 40% larger than the truth. The follow-up run then "fails to replicate", and the failure gets blamed on drift, seeds, or infrastructure.

Two consequences for planning:

- **Do not size the confirmatory fleet from an underpowered pilot's point estimate** — it is inflated by exactly this mechanism, so the resulting fleet is too small, and the error compounds. Size from the *lower* end of the effect's confidence interval, or from the smallest effect worth acting on.
- **Underpowered runs are not "some evidence".** They are a lottery ticket whose payouts are misleading. If a fleet cannot reach ~70–80% power for the effect that matters, the honest options are: reduce `sd_d`, narrow the claim, or don't run it.

## Pilot Variance Is Itself Uncertain

`sd_d` from a small pilot is a noisy estimate, and the fleet size depends on its **square**. The 95% interval for the true `sd_d` as a multiple of the observed, and the resulting range of "correct" fleet sizes for `δ = 0.012` (planned: 52):

```
 pilot n    95% CI on sd_d        implied fleet size
     5      [0.60x, 2.87x]           20  ...  407
     8      [0.66x, 2.04x]           24  ...  206
    12      [0.71x, 1.70x]           27  ...  144
    24      [0.78x, 1.40x]           32  ...   99
```

A 5-unit pilot tells you the fleet needs somewhere between 20 and 407 runs. That is not a plan.

The standard remedy is **upper-confidence-limit planning**: size the fleet using a conservative upper percentile of `sd_d` rather than its point estimate, so you are wrong in the direction that costs compute rather than the direction that wastes the whole experiment.

```
 pilot n    80% UCL factor    plan n (vs 52 at the point estimate)
     8          1.35x              92
    12          1.25x              80
    24          1.16x              68
```

Alternatives when even that is unaffordable: run an **internal pilot** (start the fleet, re-estimate `sd_d` at a pre-registered interim *without looking at the effect*, and adjust `n` — this is a blinded sample-size re-estimation and does not spend α), or plan for a fixed budget and report the MDE instead of a target power.

## Correction Costs Sample Size

Multiple-comparison control lowers the effective α, and `n` rises accordingly. At `sd_d = 0.030`, `δ = 0.012`:

```
 uncorrected      alpha = 0.050      n =  52
 2 endpoints      alpha = 0.025      n =  62
 family of 48     alpha = 0.00104    n = 112
```

Testing 48 things costs you **more than double** the fleet, for the same power on each. This is the quantitative argument for declaring one primary endpoint ([multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md)) — it is not stylistic tidiness, it is 60 training runs.

## Runnable: fleet sizing

```python
import numpy as np
from scipy.stats import t, nct, chi2, norm

def n_paired(delta, sd_d, alpha=0.05, power=0.80, exact=True, nmax=200_000):
    """Units needed for a two-sided paired test.
    delta : the smallest effect WORTH ACTING ON (not the pilot's point estimate)
    sd_d  : SD of the per-UNIT paired difference (statistical-units-and-clustering.md)
    """
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n0 = int(np.ceil((z * sd_d / delta) ** 2))
    if not exact:
        return n0
    n = max(3, n0 - 5)
    while n < nmax:
        df = n - 1
        if 1 - nct.cdf(t.ppf(1 - alpha / 2, df), df, delta / sd_d * np.sqrt(n)) >= power:
            return n
        n += 1
    raise ValueError("required n exceeds nmax -- reduce sd_d or widen delta")

def power_at(delta, sd_d, n, alpha=0.05):
    df = n - 1
    return float(1 - nct.cdf(t.ppf(1 - alpha / 2, df), df, delta / sd_d * np.sqrt(n)))

def mde(sd_d, n, alpha=0.05, power=0.80):
    """Smallest detectable effect at a FIXED fleet size. Quote this when budget
    fixes n -- it converts 'we can afford 24 runs' into a falsifiable claim."""
    lo, hi = 1e-9, 10 * sd_d
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid, sd_d, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi

def sd_ucl(sd_hat, n_pilot, conf=0.80):
    """Upper confidence limit on sd_d from a pilot. Plan with this, not sd_hat."""
    df = n_pilot - 1
    return float(sd_hat * np.sqrt(df / chi2.ppf(1 - conf, df)))

def plan(delta, sd_hat, n_pilot, alpha=0.05, power=0.80):
    ucl = sd_ucl(sd_hat, n_pilot)
    return {
        "n_at_point_estimate": n_paired(delta, sd_hat, alpha, power),
        "n_at_80pct_ucl": n_paired(delta, ucl, alpha, power),
        "sd_d_hat": sd_hat, "sd_d_ucl80": ucl,
        "recommendation": "plan to n_at_80pct_ucl; re-estimate sd_d at a blinded "
                          "interim and adjust if it comes in lower",
    }

print(plan(delta=0.012, sd_hat=0.030, n_pilot=8))   # -> 52 vs 92
print(f"MDE at n=24: {mde(0.030, 24):.4f}")          # -> 0.0179


def verdict_probabilities(n, sd_d, decision_floor, true_effects, alpha=0.05,
                          n_sim=200_000, seed=0):
    """Run the SUCCESS CRITERIA forward, before spending the budget.

    Power tells you P(reject the null). It does not tell you P(this trial returns
    an ANSWER). With ship_if = "CI lower bound > floor" and abandon_if = "CI upper
    bound < floor", a fleet can easily land in the inconclusive band almost every
    time -- and you can compute that in advance, for free.

    Do this before every confirmatory fleet. A design whose most likely output is
    "inconclusive" at the effect you actually believe in should not be run.
    """
    rng = np.random.default_rng(seed)
    crit = t.ppf(1 - alpha / 2, n - 1)
    out = {}
    for delta in true_effects:
        d = rng.normal(delta, sd_d, size=(n_sim, n))
        m = d.mean(axis=1)
        half = crit * d.std(axis=1, ddof=1) / np.sqrt(n)
        lo, hi = m - half, m + half
        ship = (m > 0) & (lo > decision_floor)
        abandon = hi < decision_floor
        out[delta] = {"ship": float(ship.mean()),
                      "abandon": float(abandon.mean()),
                      "inconclusive": float((~ship & ~abandon).mean())}
    return out

# A 20-run fleet, sd_d at the planning UCL, floor 0.020:
for delta, p in verdict_probabilities(20, 0.0307, 0.020, [0.0, 0.012, 0.019, 0.040]).items():
    print(f"true {delta:.3f} -> ship {p['ship']:.3f}  abandon {p['abandon']:.3f}  "
          f"INCONCLUSIVE {p['inconclusive']:.3f}")
```

Run on that fleet, the answer is stark:

```
true 0.000 -> ship 0.000  abandon 0.789  INCONCLUSIVE 0.211
true 0.012 -> ship 0.001  abandon 0.197  INCONCLUSIVE 0.802
true 0.019 -> ship 0.017  abandon 0.035  INCONCLUSIVE 0.948   <- the case you believe
true 0.040 -> ship 0.790  abandon 0.000  INCONCLUSIVE 0.210
```

The fleet can reliably *abandon* (if the effect is truly zero) and reliably *ship* (if it is twice what you expect). **It cannot resolve the case you actually believe you are in — 95% of the time it returns nothing.** No analysis choice fixes that, and lowering the floor widens the abandon region without making the ship region reachable. This is the calculation that tells you to spend the next week on `sd_d` rather than on runs.

## Decision Procedure

```
1. Fix the unit. Power is in units. (statistical-units-and-clustering.md)

2. Choose delta = the smallest effect WORTH ACTING ON, from the cost model
   (effect-sizes-and-cost-charged-utility.md). Do NOT use a pilot's point
   estimate -- if the pilot was underpowered, that estimate is inflated
   1.4-1.8x and your fleet will be too small.

3. Estimate sd_d: one paired difference per unit, SD across units, from a
   pilot of >= 8 units. Record n_pilot -- the uncertainty matters.

4. BEFORE sizing: can you reduce sd_d? Check the matching contract
   (common-random-numbers-and-matching.md). Halving sd_d quarters the fleet
   and improves every future experiment.

5. Compute n at the 80% upper confidence limit of sd_d, not at the point
   estimate. Add the cost of any multiple-comparison correction.

6. If n exceeds budget, do NOT run it anyway. Choose one:
   - reduce sd_d (best)
   - narrow the claim to the MDE the budget supports, and say so
   - use a blinded internal pilot to re-size mid-fleet
   - do not run

6b. Run the criteria FORWARD (verdict_probabilities) at the effects you
   actually think are plausible, including the one you believe. If the most
   likely outcome is "inconclusive" at your own best guess, the fleet cannot
   answer the question -- go back to step 4. Power alone will not reveal this.

7. Pre-register delta, sd_d, n_pilot, alpha, power, and the resulting n.
   (preregistration-and-exploratory-vs-confirmatory.md)

8. When reporting a null, ALWAYS report the MDE alongside. "No significant
   effect" without a MDE is uninterpretable.
```

## RED Scenario

> A pilot on 5 runs shows a mean paired improvement of 0.021 with `sd_d = 0.019`. The team computes `n = 8` for 80% power, runs 8 confirmatory trajectories, gets `p = 0.04`, mean 0.022, and ships it. Six weeks later a 40-run replication finds 0.004, not significant.

**The catch:** three compounding errors. (1) `sd_d = 0.019` from 5 units has a 95% CI of `[0.60×, 2.87×]` — the true value could be anywhere from 0.011 to 0.055, implying a fleet anywhere from 8 to 180. (2) `δ = 0.021` came from the same 5-run pilot, which at that size had roughly 25% power and therefore an inflated point estimate; the fleet was sized to detect an effect that was itself an artefact of low power. (3) The 8-run confirmatory result, having cleared significance at low power, is exaggerated by the Type-M mechanism — an expected 1.8× or more. The replication did not fail; the original was never a measurement.

**GREEN behaviour:** *"Nothing went wrong in the replication — the original design could not have produced a trustworthy number. Concretely: `sd_d = 0.019` estimated from 5 units carries a 95% CI of [0.011, 0.055], so 'n = 8' was one draw from a range spanning 8 to 180 runs. And δ = 0.021 was taken from the pilot's point estimate, which at ~25% power is inflated by roughly 1.8×; the replication's 0.004 is the more credible figure.*
>
> *For the next attempt: (1) set δ from the cost model — the smallest improvement that would actually change the deployment decision — not from a pilot; (2) re-estimate `sd_d` on ≥ 12 units and plan at its 80% upper confidence limit (from an 8-unit pilot that is a 1.35× inflation, taking a nominal 52 to 92); (3) before adding runs, audit the matching contract — if branches are not sharing future minibatches, `sd_d` may be 2.5× larger than it needs to be, which is worth more than any fleet increase; (4) pre-register δ, `sd_d`, α, power and n. If the resulting fleet is unaffordable, report the MDE the budget supports — at 24 runs with `sd_d = 0.030` that is 0.018 — and let the stakeholder decide whether a claim at that resolution is worth buying."*

## Cross-References

- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — power is denominated in units; branches do not count
- [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — the 0.075 → 0.030 reduction, and why it beats more runs
- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — where `δ` comes from
- [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md) — what correction costs in `n`
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — declaring the calculation before the fleet runs
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — reporting MDE alongside every null
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-12 (underpowered fleet reported as a null), AP-13 (fleet sized from an inflated pilot)
