---
name: selection-bias-and-best-of-k
description: "Use when reporting the effect of the best of several candidates, when a screening stage picks a winner, or when a promising result fails to reproduce. Covers the winner's curse, why the screen winner's measured effect is biased upward, independent audit data as the only assumption-free fix, and what analytic bias corrections actually assume."
---

# Selection Bias and Best-of-K

## Overview

**The score that made a candidate the winner is not an estimate of how good the winner is. It is a maximum, and the maximum of K noisy estimates is biased upward — by about 1.4 standard errors at K = 8, whether or not any candidate is any good.**

This is the winner's curse, and it is the reason promising screen results fail to reproduce. It is not a subtle correction: in the simulation below, eight candidates with **exactly zero** true effect produce a screen winner measuring +0.0098 with a "significant" *p*-value 18% of the time. Every bit of that is selection.

## When to Use

Use this sheet when:

- A pipeline picks the best of K candidates, configurations, seeds, prompts, checkpoints, or hyperparameter settings and then reports the winner's score.
- A result that looked strong in screening comes back weaker (or gone) on a fresh run.
- Someone reports "our best variant improved X by Y%" without saying how many variants were tried.
- You are deciding whether an admission gate needs its own data.
- You want to report an effect size for a *selected* thing rather than a pre-specified thing.

Do not use this sheet for:

- Controlling the false-positive *rate* across many tests — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md). That sheet controls whether you *declare*; this one corrects what you *report*. They are complementary and you usually need both.
- The mechanics of keeping data roles disjoint — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) (this is its L3 class).

## Core Principle

> Selection converts noise into apparent effect. Any statistic computed on the data that performed the selection is conditioned on having won, and inherits the winning margin. To estimate the winner's effect you need data that did not participate in choosing it.

## The Mechanism

With `K` candidates, each measured with standard error `SE`, the winner is the one whose noise draw was largest. Under the global null (no candidate has any true effect), the expected screen score of the winner is

```
E[ max of K standard normals ] × SE
```

which for K = 2, 4, 8, 16, 32 is roughly **0.56, 1.03, 1.43, 1.77, 2.07 SE**. The growth is logarithmic — a large K is not catastrophically worse than a moderate K, but even K = 4 buys you a full standard error of fictional effect. Note that `K` is the number of candidates you *could have picked*, not the number you report. Trying twelve prompts and reporting the best is K = 12 even if the other eleven are never mentioned.

Two properties worth internalising:

- **The bias exists with zero true effect.** It is not "regression to the mean around a real signal"; it is manufacture of signal from noise.
- **The bias is largest when candidates are near-tied.** If one candidate is clearly better, it wins on merit and the selection adds little. When all candidates are similar — the normal case in a mature system — selection dominates.

## The Numbers

Simulation: `K = 8` candidates, per-candidate `SE = 0.00694` (the running example's `sd_d / √G` at `G = 24`), 40,000 trials, three regimes.

| Estimator of the winner's true effect | A: all 8 worthless | B: one good (+0.020), 7 worthless | C: spread 0 → 0.021 |
|---|---|---|---|
| **True effect of the selected candidate** | +0.00000 | +0.01802 | +0.01796 |
| Screen score (what gets reported) | **+0.00989** *(bias +0.0099)* | +0.02038 *(bias +0.0024)* | +0.02485 *(bias +0.0069)* |
| **Independent audit estimate** | **+0.00000** *(bias ~0)* | **+0.01802** *(bias ~0)* | **+0.01793** *(bias ~0)* |
| Screen − E[max]·SE (global-null correction) | +0.00002 *(bias ~0)* | +0.01051 *(bias **−0.0075**)* | +0.01497 *(bias −0.0030)* |
| Shrinkage toward zero | +0.00000 *(bias ~0)* | +0.01039 *(bias **−0.0076**)* | +0.01314 *(bias −0.0048)* |

Read the table twice. The first read is the headline: **the screen score is badly biased and the independent audit is not, in every regime.** The second read is the warning: the two analytic corrections are excellent under the global null and *over-correct by more than the original bias* when there is a genuinely good candidate — which is exactly the case you built the pipeline to detect. A correction that erases a real +0.020 effect down to +0.010 is not a safer choice than no correction; it is a different way to be wrong.

The same simulation, viewed as decision quality:

```
K = 8 candidates, ALL with zero true effect (winner's score tested at |z| > 1.96):
  screen winner appears significant on its own SCREEN data      18.1%
  the same winner tested on independent AUDIT data               5.1%   (the nominal rate)
```

A pipeline that admits on screen evidence alone runs at a false-admission rate **more than triple** the nominal one, and no amount of care in the *test* fixes it, because the test is not the problem — the conditioning is.

## The Fix: independent audit data

**The split is the estimator.** Reserve a set of units that the screening stage never touched, and estimate the winner's effect there. This is assumption-free: it does not require knowing `K`, the noise distribution, whether effects are null, or how the selection rule worked. It works when selection was informal, iterative, or human — which analytic corrections do not.

Design requirements:

- **The audit set must be untouched by *every* selection decision**, including informal ones. If a human looked at audit-set results and then changed the candidate pool, the audit set is spent. Selection by eyeball is still selection.
- **Audit the *selected* candidate, not the pool.** The estimand is "the effect of what we chose", which is what you will deploy.
- **The audit is an estimate, not a rubber stamp.** It has its own sampling error and its own power. Auditing a winner on 4 units tells you almost nothing; size the audit set with [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md).
- **A failed audit is a result.** Record rejections; a pipeline whose audit never rejects is not auditing ([anti-pattern-catalogue.md](anti-pattern-catalogue.md), AP-08).
- **Do not re-screen on audit failure.** "The winner failed, so we audited the runner-up" turns the audit into a second screen and reinstates the curse. If you must, pre-register the fallback rule and correct for the extra look.

When the fleet cannot afford a dedicated audit role, **cross-fitting** ([grouped-splits-and-leakage.md](grouped-splits-and-leakage.md)) is the right fallback — but the selection must be refit inside each fold. Cross-fitting only the final estimate, while keeping a globally-chosen winner, leaves the bias intact.

## When Analytic Corrections Are Appropriate

They are legitimate tools with a stated domain:

- **Global-null / E[max] subtraction** — use when you genuinely believe most candidates are null (large automated sweeps, random-search pools). It answers "how much of this margin is explainable by selection alone?" Treat it as a *lower bound sanity check*, not the reported estimate. State `K` and the assumption.
- **Empirical-Bayes shrinkage** — use when you are ranking many candidates and want better *ranking* or better *average* accuracy across the pool. It is designed to minimise total squared error over many estimates, and it deliberately pulls extremes toward the middle. That property is a feature for ranking and a bug for reporting a single selected effect.
- **Conditional / truncated likelihood** (estimating the effect conditional on having been selected) — the principled parametric answer. Requires a fully specified selection rule and noise model. Real pipelines rarely have either, and the estimator is high-variance near the selection boundary.

The honest framing for a report: *"We estimate the winner's effect on held-out audit units (unbiased by construction). As a cross-check, the global-null correction to the screen score gives X, which brackets the audit estimate from below under the assumption that all candidates are null."*

## Runnable: quantify your own curse

```python
import numpy as np

def winners_curse_bias(K, se, n_sim=40000, true_effects=None, seed=0):
    """Bias of the screen score vs the true effect of the SELECTED candidate.

    true_effects=None -> global null. Pass an array of length K to explore the
    regime you actually believe you are in; the bias depends on it strongly.
    """
    rng = np.random.default_rng(seed)
    true = np.zeros(K) if true_effects is None else np.asarray(true_effects, float)
    screen = rng.normal(true, se, size=(n_sim, K))
    w = np.argmax(screen, axis=1)
    screen_score = screen[np.arange(n_sim), w]
    audit_score = rng.normal(true[w], se)          # fresh, independent data
    return {
        "true_effect_of_selected": float(true[w].mean()),
        "screen_score":            float(screen_score.mean()),
        "screen_bias":             float(screen_score.mean() - true[w].mean()),
        "audit_score":             float(audit_score.mean()),
        "audit_bias":              float(audit_score.mean() - true[w].mean()),
        "expected_max_in_SE":      float((screen_score.mean() - true[w].mean()) / se),
    }

# Before running a fleet, ask: if NOTHING works, what will the winner look like?
print(winners_curse_bias(K=8,  se=0.00694))   # ~ +0.0099, i.e. 1.43 SE of pure noise
print(winners_curse_bias(K=32, se=0.00694))   # ~ 2.07 SE -- larger pools, larger fiction

# Decision-rate view: how often does a null winner look "significant"?
rng = np.random.default_rng(2)
screen = rng.normal(0.0, 0.00694, size=(40000, 8))
winner = screen[np.arange(40000), np.argmax(screen, axis=1)]
audit  = rng.normal(0.0, 0.00694, 40000)         # fresh data for the winner
thr = 1.96 * 0.00694
print((np.abs(winner) > thr).mean())   # 0.181 -- screen "significance" under the null
print((np.abs(audit)  > thr).mean())   # 0.051 -- the honest (nominal) rate

# Pre-registration aid: the screen margin a winner must clear to be
# interesting at all, given K. Below this, you have measured nothing.
def null_screen_threshold(K, se, quantile=0.95, n_sim=200000, seed=1):
    rng = np.random.default_rng(seed)
    return float(np.quantile(np.max(rng.normal(0, se, size=(n_sim, K)), axis=1), quantile))

print(null_screen_threshold(8, 0.00694))      # 95th pct of the null winner's score
```

The `null_screen_threshold` helper belongs in your pre-registration: it converts "how many candidates are we trying?" into "how big does a margin have to be before it means anything?", *before* you see results.

## Decision Procedure

```
1. Count K honestly. Every candidate, config, prompt, seed, or checkpoint
   that COULD have been reported counts -- including abandoned ones and
   ones rejected by eye. If you cannot count K, you cannot correct for it,
   and an independent audit set becomes mandatory rather than preferred.

2. Compute the null winner's expected score (E[max of K] x SE). If your
   observed margin is not comfortably above it, stop -- there is no result
   to audit yet.

3. Estimate the winner's effect on units the screening never saw.
   This is the reported number. Everything else is a cross-check.

4. Size the audit: power it for the effect you are claiming, not for the
   screen margin (which is inflated). See power-and-sample-size...

5. Report all three: the screen score, K, and the audit estimate with its
   interval. A report that omits K cannot be evaluated by a reader.

6. Record audit rejections. An audit stage with a 0% rejection rate over
   many trials is not gating; find out why before trusting any admission.

7. Never re-screen after an audit failure without a pre-registered fallback
   rule and a corrected alpha.
```

## RED Scenario

> A team sweeps 40 candidate configurations on their evaluation set, reports *"our best configuration improves validation loss by 0.024 (p = 0.003)"*, and ships it. Three weeks later the production A/B shows no difference and they open a bug about "distribution shift".

**The catch:** K = 40. Under the global null, the expected screen score of the winner is about 2.16 SE. With this pipeline's `SE ≈ 0.0069`, that is **+0.015 of pure selection bias before any real effect exists** — well over half the reported 0.024. The *p*-value is computed as if the configuration had been specified in advance, which it was not. There is no distribution shift to debug; the number was never an estimate of anything.

**GREEN behaviour:** *"Before treating this as a shift, note that the 0.024 is a maximum over 40 candidates, and the reported p = 0.003 assumes the configuration was chosen a priori. Under the global null this sweep would be expected to produce a winner at about +0.015 by selection alone, so the production null is the predictable outcome, not an anomaly.*
>
> *What to do: (1) re-estimate the winning configuration's effect on runs the sweep never touched — that estimate is unbiased regardless of K and is the number to report; (2) if no clean units remain, re-run the top candidate on a fresh fleet sized by [power](power-and-sample-size-for-paired-designs.md) for the effect you actually expect (roughly 0.024 − 0.015 ≈ 0.009, which needs a substantially bigger fleet than the sweep did); (3) going forward, split screen and audit units up front and pre-register K, so the sweep produces a candidate rather than a claim. As a sanity check the global-null correction gives about +0.009 — but treat that as a lower bound only, since it over-corrects whenever a candidate is genuinely good."*

## Cross-References

- [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) — the screen→audit wall (L3) this sheet motivates; cross-fitting
- [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md) — controlling *declarations*, complementary to correcting *estimates*
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — sizing the audit set for the true effect, not the inflated one
- [abstention-and-calibration.md](abstention-and-calibration.md) — a screener that may pick "none of them" and how to grade it
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — declaring K and the margin threshold in advance
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-03 (winner's curse unreported), AP-08 (screening data reused as audit)
