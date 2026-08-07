---
name: horizon-choice-and-divergence-noise
description: Use when choosing how long to run matched branches before measuring, when short-horizon results contradict long-horizon ones, or when someone reports the horizon where the gap looked biggest. Covers the signal-to-divergence-noise trade-off, the interior optimum, multi-horizon endpoints without p-hacking, and horizon shopping.
---

# Horizon Choice and Divergence Noise

## Overview

**Run matched branches too briefly and the intervention has not expressed itself. Run them too long and they have drifted apart for reasons that have nothing to do with the intervention. The best evaluation horizon is an interior optimum, and finding it *after* seeing results is p-hacking with a technical-sounding name.**

Horizon is the parameter that most directly controls your statistical power, and it is usually chosen by convention or convenience. In the model below, the difference between a well-chosen and a badly-chosen horizon is a fleet of 26 runs versus 222.

## When to Use

Use this sheet when:

- Deciding how many steps, epochs, or episodes a branch runs before measurement.
- Short-horizon and long-horizon results disagree about a candidate.
- Someone proposes reporting "the horizon where the effect is clearest".
- Branch variance grows visibly with run length and you need to know whether to shorten.
- You want multiple horizons in the analysis without inflating the family.

Do not use this sheet for:

- What keeps branches matched in the first place — [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md).
- The family-size arithmetic once horizons are endpoints — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md).

## Core Principle

> Choose the horizon that maximises signal-to-divergence-noise, estimate it on pilot units, and freeze it before the confirmatory fleet. The horizon is a design parameter with an optimum — not a reporting choice with a best case.

## The Trade-off

Two quantities move in opposite directions as the horizon `H` grows:

- **Signal.** The intervention's effect needs time to express. It typically rises and then **saturates** — after the mechanism has done its work, more steps add nothing.
- **Divergence noise.** Two branches that start identical drift apart. Even with perfect common random numbers, a small difference at step 0 compounds through the optimisation dynamics, so `sd_d` **grows without bound** in `H`.

Signal saturates; noise does not. The ratio therefore peaks and then declines — there is always an interior optimum, and the only question is where.

An illustrative model (signal `a(1 − e^{−H/τ})` with `a = 0.016, τ = 2500`; `sd_d` growing as `√H` from a floor of 0.006):

```
      H    effect     sd_d      d_z    runs for 80% power
    250    0.0015   0.0081    0.189          222
    500    0.0029   0.0097    0.299           90
   1000    0.0053   0.0123    0.428           45
   2000    0.0088   0.0163    0.539           29
   3000    0.0112   0.0196    0.571           27
   3500    0.0121   0.0212    0.571           26   <- optimum
   5000    0.0138   0.0248    0.558           28
   8000    0.0153   0.0310    0.495           34
  12000    0.0159   0.0377    0.421           47
  20000    0.0160   0.0485    0.330           74
  40000    0.0160   0.0683    0.234          145
```

Three things to take from this table:

1. **The optimum is interior and the penalty is asymmetric-looking but real on both sides.** At `H = 250` you need 222 runs because there is nothing to see yet; at `H = 40000` you need 145 because the branches have wandered. The middle costs 26.
2. **The optimum is broad.** Anywhere from 2000 to 5000 costs within 10% of the best. Do not over-tune — pick a round number in the plateau, which also makes the choice easier to defend as non-data-driven.
3. **The largest effect is not at the best horizon.** The raw effect at `H = 40000` (0.0160) is 32% larger than at the optimum (0.0121), and needs 5.6× the fleet to establish. **"The effect is biggest at the long horizon" is compatible with the long horizon being the worst place to measure it.**

That last point is the one that produces bad decisions, because the intuition ("bigger effect = easier to detect") is wrong whenever noise grows faster than signal.

## Horizon Shopping

Evaluating at several horizons and reporting the one with the smallest *p*-value is a textbook researcher degree of freedom ([preregistration…](preregistration-and-exploratory-vs-confirmatory.md)). It has two costs that compound:

- **Family inflation.** Five horizons is five tests; the reported one is a maximum over them, so the nominal α is wrong.
- **Effect inflation.** The horizon was selected for having the largest gap, so the estimate carries the same winner's-curse bias as any selected quantity ([selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)).

The tell in a codebase: an analysis script whose `HORIZON` constant was edited after the fleet finished, or a results table with a `best_horizon` column. The tell in a write-up: a horizon that appears nowhere in the design document.

**Horizon shopping is not the same as multi-horizon analysis.** Looking at several horizons is good practice — the trajectory of the effect over `H` is informative and often more useful than any single point. The rule is only that you cannot *select* among them post hoc and then report a nominal `p`.

## Multi-Horizon Endpoints Done Honestly

Four legitimate designs, in rough order of preference:

- **One primary horizon, secondary horizons descriptive.** Declare `H*` in the pre-registration; report the others as estimates and intervals with no inferential claim. Simplest, keeps `m = 1`, and covers most cases.
- **Pre-declared composite.** Define the endpoint as a weighted average across horizons (e.g. `0.5·u(H=2000) + 0.5·u(H=8000)`) with weights fixed in advance. One test, and it captures "helps soon and still helps later".
- **Hierarchical gatekeeping.** Order horizons by importance; test at full α in sequence, stopping at the first non-rejection. Controls FWER without splitting α.
- **Correct across horizons.** Test all of them and apply Holm. Honest but weak, because horizon estimates are strongly correlated and the correction does not exploit that.

Whichever you pick, **report the effect-versus-horizon curve with intervals.** It is the single most informative plot in a counterfactual write-up: it shows the saturation point, shows where divergence takes over, and lets a reader see that the declared horizon was not cherry-picked. It also makes the "helps early, regresses late" pattern visible, which a single horizon can hide entirely — and that pattern is usually the most important thing a trial can discover.

## Estimating the Optimum

Do this on **pilot units, before the confirmatory fleet** — the horizon is a design parameter fit on pilot data, and it obeys the same freezing rules as thresholds and cost weights.

1. Run a small pilot (≥ 8 units) with branches carried out to the longest horizon you would consider, checkpointing the metric at a grid of intermediate horizons. One fleet gives you the whole curve.
2. At each horizon, compute the per-unit paired difference and its `sd_d` across units.
3. Plot `d_z(H) = effect(H) / sd_d(H)` and the implied `n(H)`.
4. Choose a round number in the plateau of the `d_z` curve.
5. Freeze it. Record the curve and the choice in the pre-registration.

Two practical cautions. The pilot's `d_z` curve is itself noisy — with 8 units, `sd_d` carries a wide interval ([power](power-and-sample-size-for-paired-designs.md)), so prefer the plateau's centre over its argmax. And **divergence growth is a property of your harness, not of nature**: if `sd_d` grows faster than `√H`, suspect that common random numbers are decaying — a stream that desynchronises partway through looks exactly like accelerated divergence ([common-random-numbers-and-matching.md](common-random-numbers-and-matching.md)).

## Runnable: horizon selection from pilot data

```python
import numpy as np
from scipy.stats import t, nct

def n_paired(delta, sd_d, alpha=0.05, power=0.80, nmax=200_000):
    n = 3
    while n < nmax:
        df = n - 1
        if 1 - nct.cdf(t.ppf(1 - alpha / 2, df), df, delta / sd_d * np.sqrt(n)) >= power:
            return n
        n += 1
    return nmax

def horizon_curve(diffs_by_horizon):
    """diffs_by_horizon: {H: array of ONE paired difference per pilot unit}.

    Returns the signal-to-divergence-noise curve. Run this on PILOT units and
    freeze the choice -- computing it after the confirmatory fleet and picking
    the best horizon is horizon shopping.
    """
    rows = []
    for H in sorted(diffs_by_horizon):
        x = np.asarray(diffs_by_horizon[H], float)
        eff, sd = float(x.mean()), float(x.std(ddof=1))
        rows.append({"horizon": H, "effect": eff, "sd_d": sd,
                     "d_z": eff / sd if sd > 0 else np.nan,
                     "n_for_80pct": n_paired(abs(eff), sd) if sd > 0 and eff != 0 else None,
                     "n_pilot": len(x)})
    return rows

def recommend_horizon(rows, plateau_tol=0.10):
    """Pick the CENTRE of the plateau within tol of peak d_z, not the argmax.
    The argmax of a noisy pilot curve is itself a selected maximum."""
    dz = np.array([r["d_z"] for r in rows], float)
    peak = np.nanmax(dz)
    inside = [r["horizon"] for r, v in zip(rows, dz) if v >= peak * (1 - plateau_tol)]
    return {"plateau": (min(inside), max(inside)),
            "recommended": int(np.median(inside)),
            "peak_d_z": float(peak),
            "note": "round this to a defensible number and freeze it in the pre-registration"}

def arm_correlation_by_horizon(pairs_by_horizon):
    """The most direct evidence that matching is decaying with horizon.

    pairs_by_horizon: {H: (control_values, treatment_values)} at the CELL level
    (one entry per matched branch pair), not aggregated.

    Var(D) = 2*sigma^2*(1 - rho), so the paired variance reduction you are
    actually getting is 1/(1-rho). Watching rho fall as H grows is the mechanism
    behind the d_z curve turning over -- and it distinguishes 'the signal
    saturated' from 'the branches stopped being matched', which the effect
    column alone cannot.
    """
    out = []
    for H in sorted(pairs_by_horizon):
        c, t = (np.asarray(x, float) for x in pairs_by_horizon[H])
        rho = float(np.corrcoef(c, t)[0, 1])
        out.append({"horizon": H, "arm_correlation": rho,
                    "paired_variance_reduction": (1.0 / (1.0 - rho)) if rho < 1 else np.inf})
    if len(out) > 1 and out[-1]["arm_correlation"] < out[0]["arm_correlation"] - 0.10:
        out.append({"note": "arm correlation is decaying with horizon -- the longer "
                            "horizon is buying a bigger raw effect with strictly less "
                            "evidence. Check CRN before choosing the long horizon."})
    return out


def divergence_growth_check(rows):
    """sd_d should grow roughly as sqrt(H). Much faster growth suggests the
    shared random streams are desynchronising rather than genuine divergence."""
    H = np.array([r["horizon"] for r in rows], float)
    sd = np.array([r["sd_d"] for r in rows], float)
    ok = (H > 0) & (sd > 0)
    slope = np.polyfit(np.log(H[ok]), np.log(sd[ok]), 1)[0]
    return {"log_log_slope": float(slope),
            "expected_approx": 0.5,
            "verdict": "consistent with genuine divergence" if slope < 0.75 else
                       "sd_d grows too fast -- check CRN stream alignment "
                       "(common-random-numbers-and-matching.md)"}
```

## Decision Procedure

```
1. Decide the longest horizon worth considering from the DOMAIN: how long
   must an effect persist to matter for the decision? That is the ceiling,
   not the measurement point.

2. Pilot >= 8 units to that ceiling, checkpointing at a grid of horizons.
   One fleet, whole curve.

3. Compute effect(H), sd_d(H), d_z(H), n(H) per horizon (horizon_curve).

4. Run divergence_growth_check AND arm_correlation_by_horizon. If sd_d grows
   much faster than sqrt(H), or the arm correlation is falling, fix CRN before
   choosing anything -- you are measuring a broken harness. The correlation is
   the more direct evidence: it separates "the signal saturated" from "the
   branches stopped being matched".

5. Choose the CENTRE of the d_z plateau, rounded. Not the argmax.

6. Freeze it in the pre-registration as the single primary horizon.

7. Report other horizons descriptively, with the effect-vs-horizon curve
   and intervals. No inferential claims on secondary horizons unless you
   pre-declared a composite, a gatekeeping order, or a correction.

8. If short and long horizons genuinely disagree in SIGN, that is the
   finding -- report it as "helps early, regresses late" with both
   estimates. Do not resolve it by choosing the horizon you prefer.
```

## RED Scenario

> A trial evaluates candidates at H ∈ {1k, 5k, 20k}. The write-up reports: *"At the 20k horizon the candidate improves utility by 0.016 (p = 0.04, n = 30)."* The design document specifies 5k. A results notebook contains a cell computing all three horizons, run before the write-up.

**The catch:** horizon shopping. Three horizons were evaluated, the design named 5k, and the reported one is 20k — selected after the fact. Two independent inflations: the family is at least 3 (nominal `p = 0.04` becomes roughly 0.12 under Holm), and the estimate is a maximum over horizons and therefore biased upward. Worse, in a system where divergence noise grows with `H`, the long horizon is likely to be the *least* powerful place to measure — the larger effect there is bought at a variance cost that makes the fleet requirement higher, not lower.

**GREEN behaviour:** *"The design named 5k as the primary horizon; the write-up reports 20k. Unless there is a documented pre-registered amendment, that is a post-hoc selection over three horizons and the nominal p = 0.04 does not hold — Holm across three puts it near 0.12.*
>
> *Two things to do. First, report the pre-registered 5k result as the confirmatory number, whatever it says, and present 20k as exploratory with its interval. Second — and more useful — plot effect and `sd_d` against horizon with intervals. If `sd_d` is growing with `H`, the 20k effect being larger does not make it more detectable; on the model shape typical of these systems, a 32% larger raw effect at a long horizon can require 5× the fleet. That plot will also tell you whether the true optimum sits near 3k, which would mean the design's 5k was close and 20k was the worst of the three.*
>
> *Also run the divergence-growth check: if `sd_d` grows faster than about √H, the shared random streams are desynchronising and the long-horizon numbers are measuring the harness, not the candidate. Going forward, choose the horizon from a pilot `d_z` curve, freeze it, and report the full curve so the choice is auditable."*

## Cross-References

- [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — matching decays with horizon; the divergence-growth check
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — `d_z(H)` converts directly into fleet size
- [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md) — horizons as a family dimension
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — freezing the horizon before the fleet
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — a selected horizon is a selected maximum
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — the effect-vs-horizon curve as a reporting artifact
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-07 (horizon shopping)
