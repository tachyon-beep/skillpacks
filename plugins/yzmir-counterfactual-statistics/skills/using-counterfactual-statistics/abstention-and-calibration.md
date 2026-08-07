---
name: abstention-and-calibration
description: Use when evaluating a judge or screener that may decline to act, when choosing an admission threshold, or when a confidence score is used as if it were a probability. Covers no-op precision and recall, false-intervention rate, expected regret as the threshold objective, reliability diagrams and ECE, and freezing calibration before confirmatory runs.
---

# Abstention and Calibration

## Overview

**A judge that can say "do nothing" is not a classifier with an extra class — it is a decision rule whose value is measured in regret against the oracle, not in accuracy. And the threshold that decides when it acts is a *fitted parameter*: fit it on validation units, freeze it, and never let it touch the units you will report.**

Abstention changes what "good" means. A screener that admits everything can look accurate while destroying value; a screener that admits nothing has perfect precision and produces zero benefit. This sheet gives the metrics that distinguish them and the procedure for setting the threshold honestly.

## When to Use

Use this sheet when:

- An automated gate decides whether to apply a candidate at all, and "none of them" is a legitimate output.
- You need to choose or defend an admission threshold, confidence cut-off, or uncertainty band.
- A model emits a confidence score that downstream code treats as a probability.
- Someone reports "the screener is 87% accurate" for a system whose base rate of useful interventions is 15%.
- You are comparing a cheap field screener against expensive ground-truth evaluation.

Do not use this sheet for:

- The bias in the *selected* candidate's measured effect — [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md).
- Which data the threshold may be fit on — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) sets the rule; this sheet applies it.
- Whether the underlying utility is worth acting on — [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md).

## Core Principle

> Grade an abstaining judge against two baselines it must beat: **always act** and **never act**. If it cannot beat both, it is not adding decision value — it is adding cost and the appearance of rigour.

## The Metrics

Let `u` be the true cost-charged utility of the best available candidate at a decision point ([effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)). Acting is correct when `u > 0`; abstaining is correct when `u ≤ 0`.

| Metric | Definition | What it catches |
|---|---|---|
| **No-op precision** | of the decisions where we abstained, the fraction where abstaining was right | A gate that refuses good candidates |
| **No-op recall** | of the decisions where abstaining was right, the fraction where we abstained | A gate that lets harmful candidates through |
| **False-intervention rate** | of the decisions where we acted, the fraction where `u ≤ 0` | The rate at which the system does damage |
| **Realised utility** | mean of `u` where we acted, 0 where we abstained | The only number that pays rent |
| **Expected regret** | mean of `max(u,0) − (u if acted else 0)` | Distance from the oracle; the threshold objective |
| **Abstention rate** | fraction of decisions where we did nothing | Sanity: near 0% or near 100% both signal a broken gate |

Report all of them. Accuracy alone is uninformative: with a 15% base rate of useful interventions, "always abstain" scores 85% accurate and delivers nothing.

**Regret is the objective; the rest are diagnostics.** Regret is what you lose relative to a perfect judge, it is in the same units as utility, and it is minimised at the threshold you actually want.

## The Numbers

A screener on 4,000 decision points. Utility `u ~ N(0.002, 0.020)`, so acting is correct 54.4% of the time. The screener predicts `u` with noise and emits a probability.

```
  thr  admit%   noopP   noopR  falseInt    regret   realised
 0.30   63.5%   0.889   0.711     0.208   0.00118   +0.00818
 0.40   58.5%   0.857   0.779     0.172   0.00101   +0.00834
 0.50   53.7%   0.825   0.837     0.138   0.00099   +0.00837   <- min regret
 0.60   48.9%   0.787   0.882     0.110   0.00108   +0.00828
 0.70   44.3%   0.749   0.916     0.087   0.00126   +0.00810
 0.80   37.5%   0.694   0.950     0.061   0.00171   +0.00765
 0.90   29.3%   0.634   0.982     0.027   0.00247   +0.00688

 oracle (act iff u>0)   realised +0.00936   regret 0
 always act             realised +0.00219
 never act              realised +0.00000
```

Three things to read off this table:

1. **Abstention is where the value is.** Always-acting realises +0.00219; the screener at its best threshold realises +0.00837 — **89% of the oracle's +0.00936, versus 23% for always-acting.** Nearly all the benefit comes from *declining*, not from picking well. That is the usual shape, and it is why a system without a no-op option leaves most of its value on the floor.
2. **Tightening the threshold looks safer and is not.** Going from 0.50 to 0.90 cuts the false-intervention rate from 13.8% to 2.7% — a number that reads well in a review — while regret rises 2.5× and realised utility falls by 18%. **A gate tuned to minimise false interventions will converge on never acting.** Optimise regret; report false-intervention rate as a constraint, not an objective.
3. **The optimum is flat.** Regret between thresholds 0.40 and 0.60 varies by 9%. Do not over-tune; pick a round number in the flat region and note the flatness, or you are fitting the validation set's noise.

## Calibration

A threshold on a probability is only meaningful if the probability means something. **Calibration** means: among decisions assigned probability ≈ 0.7, about 70% should turn out to be right.

Measure it two ways:

- **Reliability diagram** — bin predictions, plot mean predicted vs empirical rate. The shape names the pathology: an S-curve pulled toward 0 and 1 is an overconfident model; a flat curve is a model with no discrimination.
- **Expected calibration error (ECE)** — the bin-count-weighted mean absolute gap. One number, useful for tracking; it hides direction and shape, so never report it without the diagram.

The example screener, held-out half: **ECE = 0.0666**. Its reliability curve shows the classic overconfidence signature — predictions near 0.15 come true 30% of the time, predictions near 0.85 come true 76% of the time. Temperature scaling with a single parameter fit on the *other* half gives `T = 1.51` and **ECE = 0.0248**, a 2.7× improvement for one scalar.

Calibration methods, in ascending order of data appetite:

| Method | Parameters | Use when |
|---|---|---|
| **Temperature scaling** | 1 | Default. Fixes over/under-confidence without touching ranking. Cheap, hard to overfit. |
| **Platt scaling** | 2 | Also corrects a systematic offset (base-rate shift). |
| **Isotonic regression** | non-parametric | Non-monotone miscalibration, and you have hundreds of validation units to spare. Overfits readily at small `G`. |

Two rules that prevent the common errors:

- **Calibration does not change ranking** (temperature and Platt are monotone). If your problem is that the screener ranks candidates badly, calibration will not help — that is a modelling problem.
- **Fit calibration on units, not rows.** A calibration set drawn by row from clustered data is subject to the same sibling leakage as any other split ([grouped-splits-and-leakage.md](grouped-splits-and-leakage.md), L1) and will report an ECE that is too good.

## Freeze Before the Confirmatory Run

The threshold and the calibration map are **fitted parameters of the decision system**. Everything in [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) applies to them:

- Fit on **screen/validation** units. Never on audit or report units.
- **Freeze both before the confirmatory run begins**, and record them in the pre-registration ([preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md)) with their fitted values and the data they came from.
- Re-tuning the threshold after seeing confirmatory results converts the confirmatory run into an exploratory one. If you must re-tune, say so and label the result exploratory — that is a legitimate move; presenting it as confirmatory is not.
- **Recalibration is a change to the system.** Version it, date it, and record which units it was fit on. A calibration map silently refreshed on production data will eventually be fit on the very units used to report.

## Runnable: threshold selection and calibration

```python
import numpy as np
from scipy.optimize import minimize_scalar

def abstention_report(u, p, thresholds=np.arange(0.05, 1.0, 0.05)):
    """u: true cost-charged utility per decision. p: screener probability.
    Acting is correct iff u > 0; the no-op option has utility exactly 0."""
    rows = []
    for t in thresholds:
        act = p >= t
        tp, fp = ((u > 0) & act).sum(), ((u <= 0) & act).sum()
        fn, tn = ((u > 0) & ~act).sum(), ((u <= 0) & ~act).sum()
        rows.append({
            "threshold": float(t), "abstain_rate": float(1 - act.mean()),
            "noop_precision": float(tn / max(tn + fn, 1)),
            "noop_recall": float(tn / max(tn + fp, 1)),
            "false_intervention_rate": float(fp / max(act.sum(), 1)),
            "realised_utility": float(np.where(act, u, 0.0).mean()),
            "regret": float((np.maximum(u, 0) - np.where(act, u, 0.0)).mean()),
        })
    best = min(rows, key=lambda r: r["regret"])
    return rows, best, {
        "oracle": float(np.maximum(u, 0).mean()),
        "always_act": float(u.mean()),
        "never_act": 0.0,
        "value_captured": float(best["realised_utility"] / max(np.maximum(u, 0).mean(), 1e-12)),
    }


def temperature_scale(logits_fit, y_fit):
    """Single-parameter calibration. Fit on VALIDATION units; apply everywhere."""
    def nll(T):
        q = np.clip(1 / (1 + np.exp(-logits_fit / T)), 1e-9, 1 - 1e-9)
        return -(y_fit * np.log(q) + (1 - y_fit) * np.log(1 - q)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def ece(p, y, bins=10):
    """Expected calibration error. Always look at the reliability bins too."""
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability_bins(p, y, bins=10):
    """Returns (mean predicted, empirical rate, count) per bin -- the diagram."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        out.append((float(p[m].mean()) if m.sum() else None,
                    float(y[m].mean()) if m.sum() else None, int(m.sum())))
    return out
```

Note `value_captured` in the summary: the fraction of oracle utility realised. It is the single most communicative number about an abstaining judge, and it is the one that makes "89% of oracle versus 23% for always-act" legible to a reader who does not want a table.

## Decision Procedure

```
1. Define utility so that abstaining scores exactly 0. Acting is correct
   iff u > 0. If your utility does not have a natural zero, fix that first
   (effect-sizes-and-cost-charged-utility.md) -- abstention is undefined
   without an anchor.

2. Obtain ground-truth u on a set of VALIDATION units (expensive full
   evaluation). This is what the cheap screener is calibrated against.

3. Calibrate: fit temperature (or Platt) on validation units. Check ECE
   AND the reliability diagram before and after. Fit on units, not rows.

4. Sweep the threshold on validation units; choose by MINIMUM REGRET, not
   by minimum false-intervention rate. Note the width of the flat region
   and pick a round number inside it.

5. Compare to both baselines. Report value_captured vs oracle, and the
   always-act and never-act numbers. If the judge does not beat both,
   it is not earning its cost.

6. FREEZE threshold + calibration map. Record values, fit date, and the
   unit ids they were fit on, in the pre-registration.

7. On the confirmatory/report units, apply the frozen rule and report the
   full metric set. Do not re-tune. If you do, relabel as exploratory.

8. Monitor drift: track abstention rate and ECE over time. A drifting
   abstention rate with a frozen threshold means the input distribution
   moved -- that is a recalibration trigger and a new version, not a
   silent refit.
```

## RED Scenario

> A team ships a field screener and reports: *"91% accuracy, and we tuned the admission threshold to 0.9 to keep false interventions under 3%."* The system's abstention rate in production is 71%.

**The catch:** three problems stacked. (1) Accuracy is uninformative here — with this base rate, always-abstaining scores in the mid-80s. (2) The threshold was optimised against the wrong objective: minimising false interventions monotonically pushes the threshold up, and its limit is "never act". At 0.90 this screener realises +0.00688 against a best-achievable +0.00837 — it is leaving 18% of its own capability unused and 26% of the oracle's. (3) A 71% abstention rate against a 46% correct-abstention base rate says the gate is refusing a large number of genuinely useful interventions; no-op recall will be near 0.98 and no-op precision near 0.63.

**GREEN behaviour:** *"Accuracy and false-intervention rate are the wrong pair of numbers to tune on — minimising false interventions has 'never act' as its optimum, and the 71% abstention rate suggests you are close to it. Re-tune against **expected regret** on validation units: on data of this shape the regret minimum sits near 0.50, realising +0.00837 (89% of oracle) against +0.00688 at 0.90 (74%). If a 3% false-intervention cap is a real business constraint rather than an inherited default, keep it — but state it as a constraint and report the regret cost of honouring it, so the trade is visible.*
>
> *Also: check calibration before touching the threshold. If the screener is overconfident, the number 0.9 does not mean 90% — on the example screener, ECE is 0.067 raw and 0.025 after a single temperature parameter, and every threshold moves once probabilities mean what they say. Fit calibration and threshold on validation units only, then freeze both and record them in the pre-registration; re-tuning after seeing the report units would make this an exploratory result, not a confirmatory one. Finally, report the always-act and never-act baselines alongside — without them a reader cannot tell whether the gate is adding decision value at all."*

## Cross-References

- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — defining the `u` this sheet thresholds, with a real zero
- [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) — the threshold and calibration map are fitted parameters and obey the wall
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — freezing before confirmatory runs
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — the winner the gate is judging is itself a maximum
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — abstention rate and false-intervention rate as first-class reported numbers
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-04 (threshold tuning on report data), AP-11 (accuracy on an abstaining judge)
