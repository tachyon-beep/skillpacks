---
name: multiple-comparisons-and-sequential-testing
description: "Use when a trial produces many p-values across candidates, horizons, and metrics, when someone peeks at results mid-run and stops early, or when choosing between familywise and false-discovery control. Covers family definition, Bonferroni/Holm/BH, alpha spending for interim looks, and why peeking without a boundary invalidates the nominal alpha."
---

# Multiple Comparisons and Sequential Testing

## Overview

**A counterfactual trial does not produce one test. It produces candidates × horizons × metrics × interim looks, and every one of them is a chance to be wrong. At 48 uncorrected tests, the probability of at least one false positive is 91.5% — you are not testing a hypothesis, you are harvesting noise.**

Two distinct inflations are at work: *breadth* (many things tested at once) and *depth in time* (the same thing tested repeatedly as data accumulates). They need different machinery. This sheet covers both, and — as importantly — how to shrink the family so you need less correction in the first place.

## When to Use

Use this sheet when:

- A results table has more than a handful of *p*-values.
- You test the same intervention at several horizons or on several metrics.
- Someone checks the fleet's results before it finishes and proposes stopping early.
- A dashboard recomputes significance continuously as runs land.
- You need to choose between controlling *any* false positive and controlling the *proportion* of them.

Do not use this sheet for:

- Correcting the *estimate* of a selected winner — [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md). That sheet fixes the reported magnitude; this one fixes the error rate of declarations. A pipeline that selects *and* declares needs both.
- Deciding which endpoint is primary in the first place — [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md).

## Core Principle

> Correction is arithmetic; the hard part is honestly declaring the family. The family is every test you *could have reported*, including the ones you looked at and discarded. If the family is under-declared, no correction procedure saves you.

## How Bad Is Uncorrected?

The running example's trial shape: 8 candidates × 2 horizons × 3 metrics = 48 tests.

```
family size m = 48, alpha = 0.05, all nulls true:
  P(at least one false positive) = 1 - 0.95^48 = 91.5%
```

You will almost certainly find something. And because the finding is selected for being extreme, its effect size is also inflated ([selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)) — the two failures compound into a result that is both spurious and impressively large.

## Familywise vs False Discovery

| | **FWER** (Bonferroni, Holm) | **FDR** (Benjamini–Hochberg) |
|---|---|---|
| Controls | P(≥1 false positive) ≤ α | Expected *proportion* of false positives among rejections ≤ α |
| Use when | One wrong call is expensive: an admission gate, a safety claim, a go/no-go decision, a headline | Many findings will be followed up anyway: exploratory screening, feature triage, candidate shortlisting |
| Cost | Conservative; power drops fast with `m` | More discoveries, some of which are wrong by design |
| Assumption | None (Bonferroni), none (Holm) | Independence or positive dependence (BH); use BY for arbitrary dependence at a real power cost |

Choose by **what a false positive costs you**. If a false positive causes an intervention to be admitted into a live system, you want FWER on that decision. If a false positive causes one more candidate to enter a cheap follow-up queue, FDR is the better trade.

**Holm dominates Bonferroni**: it controls FWER just as strictly and rejects at least as much, always. There is no reason to use plain Bonferroni for testing — though Bonferroni is still useful as a back-of-envelope for *planning*, because `α/m` is easy to reason about.

The three procedures on ten *p*-values from a single trial:

```
raw    0.001  0.008  0.021  0.033  0.041  0.049  0.12  0.31  0.44  0.77
Bonf   0.010  0.080  0.210  0.330  0.410  0.490  1.00  1.00  1.00  1.00   -> 1 rejection
Holm   0.010  0.072  0.168  0.231  0.246  0.246  0.48  0.93  0.93  0.93   -> 1 rejection
BH     0.010  0.040  0.070  0.082  0.082  0.082  0.17  0.39  0.49  0.77   -> 2 rejections
```

Six raw *p*-values are below 0.05; one or two survive correction. That gap is the size of the problem.

## Shrink the Family First

Correction is the last resort, not the first move. Every one of these reduces `m` before any adjustment:

- **One primary endpoint.** Declare a single metric at a single horizon as the confirmatory test. Everything else is secondary and reported without inferential claims. This takes `m` from 48 to 1 for the decision that matters. It is by far the highest-leverage move in this sheet.
- **Composite or hierarchical endpoints.** If three metrics all matter, combine them into one pre-declared utility ([effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)) rather than testing three.
- **Hierarchical (gatekeeping) testing.** Order endpoints by importance; test in sequence at full α and stop at the first non-rejection. Controls FWER with no α splitting at all — powerful when the ordering is genuinely justified and declared in advance.
- **Test the pool, not each candidate.** Often the question is "did *any* candidate help?" — one test on the per-unit utility of the *selected* candidate, not `K` tests. The selection is then handled as a selection problem, which is the correct framing.
- **Do not test what you will not act on.** A *p*-value attached to a metric nobody would change a decision over is pure family inflation.

## Sequential Testing and Interim Looks

Looking at accumulating data and stopping when the result is significant inflates the error rate, because you get one chance to be wrong per look, and the looks are chosen adversarially by the data.

```
3 interim looks + 1 final (equally spaced), each at nominal alpha = 0.05:
  true type-I error  12.6%  (upper bound 1 - 0.95^4 = 18.5%; correlated looks reduce it)
```

Continuous monitoring is worse: with unlimited looks and a fixed nominal α, a null effect is *guaranteed* to cross the threshold eventually. "We'll just stop when it turns significant" is a procedure with a type-I error rate of 1.

The fix is an **alpha-spending function**: decide in advance how much of your total α each look may spend, as a function of the information fraction. The O'Brien–Fleming shape is the standard default — it spends almost nothing early (so early stopping requires overwhelming evidence) and preserves nearly the full α for the final analysis:

```
info fraction   cumulative alpha spent   nominal z boundary
     0.25              0.0001                   3.92
     0.50              0.0056                   2.78
     0.75              0.0236                   2.30
     1.00              0.0500                   2.04
```

The final boundary is 2.04 rather than 1.96 — the price of three interim looks under this shape is small at the end, which is why it is the default. Pocock's shape (constant boundary) stops earlier more readily but costs real α at the final look; choose it only when early stopping has high value.

One implementation error is common enough to name: reading each look's boundary off the **cumulative** spend with a plain normal quantile (`norm.ppf(1 - spent/2)`). That treats every look as a fresh single test, emits a final boundary of 1.96, and realises an overall α of ≈ 0.059 instead of 0.050. Each boundary must be solved so that the probability of crossing at *this* look, having crossed at *no earlier* one, equals the spending *increment* — the recursion in the runnable below does exactly that.

Practical guidance for ML fleets:

- **Pre-register the number and timing of looks, and the spending function**, before the first result lands. Looks scheduled after seeing data are not interim analyses; they are peeking.
- **Information fraction, not calendar time.** With `G` planned units, the fraction is `units_completed / G`, not `days_elapsed / days_planned`.
- **Futility stopping is cheaper than efficacy stopping.** Stopping early because the effect is clearly *not* there costs little α (you are not making a positive claim) and saves a lot of compute. Pre-register a futility boundary — most fleets should have one.
- **A dashboard that recomputes significance on every arriving run is continuous monitoring.** Either put a boundary on it, or label it explicitly as a monitoring view with no inferential status. The second option is usually right: let people watch the estimate and its interval, and reserve the *test* for the pre-registered analysis.
- **Sequential methods that permit truly continuous monitoring exist** (always-valid *p*-values, confidence sequences, mixture SPRTs). They are the right tool if you genuinely need anytime-valid inference; they trade a constant factor of power for the freedom to look whenever you like. Adopt them deliberately, not as a retrofit.

## Runnable: corrections and spending boundaries

```python
import numpy as np
from scipy.stats import norm

def holm(p):
    """Holm-Bonferroni step-down adjusted p-values. FWER control, no assumptions.
    Dominates Bonferroni -- always prefer it for testing."""
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty_like(p), 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

def benjamini_hochberg(p):
    """BH step-up adjusted p-values (q-values). FDR control under independence
    or positive dependence. Use when findings feed a follow-up queue."""
    p = np.asarray(p, float)
    n, order, adj, prev = len(p), np.argsort(p), np.empty_like(p), 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, n / rank * p[idx])
        adj[idx] = prev
    return adj

def obf_alpha_spent(info_fraction, alpha=0.05):
    """O'Brien-Fleming (Lan-DeMets) cumulative alpha at a given information
    fraction. Pre-register the LOOK SCHEDULE, then read boundaries off this."""
    f = np.clip(np.asarray(info_fraction, float), 1e-9, 1.0)
    return 2 - 2 * norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(f))

def obf_boundaries(look_fractions, alpha=0.05, n_grid=2001, zmax=8.0):
    """Nominal two-sided z boundary at each planned look, by the Lan-DeMets
    recursion: each boundary is solved so that the probability of crossing at
    THIS look, having crossed at no earlier one, equals the spending increment.

    Do NOT shortcut this with norm.ppf(1 - spent/2) on the cumulative spend --
    that treats every look as a fresh single test, emits a final boundary of
    1.96 instead of 2.04, and realises alpha ~= 0.059 rather than 0.050.
    """
    f = np.asarray(look_fractions, float)
    spent = obf_alpha_spent(f, alpha)
    incr = np.diff(np.concatenate([[0.0], spent]))
    out, dens, s, t_prev = [], None, None, 0.0
    for t_k, a_k in zip(f, incr):
        s_k = np.linspace(-zmax, zmax, n_grid) * np.sqrt(t_k)  # grid over S(t_k)
        ds = s_k[1] - s_k[0]
        if dens is None:
            d_k = norm.pdf(s_k, scale=np.sqrt(t_k))
        else:                      # convolve survivors with the new increment
            d_k = norm.pdf(s_k[:, None] - s[None, :],
                           scale=np.sqrt(t_k - t_prev)) @ dens * (s[1] - s[0])
        lo, hi = 0.0, zmax         # bisect: outside-mass == this look's spend
        for _ in range(60):
            mid = (lo + hi) / 2
            if (d_k * (np.abs(s_k) > mid * np.sqrt(t_k))).sum() * ds > a_k:
                lo = mid
            else:
                hi = mid
        z_k = (lo + hi) / 2
        out.append((float(t_k), float(spent[len(out)]), float(z_k)))
        dens = d_k * (np.abs(s_k) <= z_k * np.sqrt(t_k))       # survivors only
        s, t_prev = s_k, t_k
    return out

def family_size_warning(n_candidates, n_horizons, n_metrics, alpha=0.05):
    m = n_candidates * n_horizons * n_metrics
    return {"family_size": m,
            "p_any_false_positive": float(1 - (1 - alpha) ** m),
            "bonferroni_alpha": alpha / m,
            "advice": "declare ONE primary endpoint" if m > 5 else "family is small; correct and move on"}

print(family_size_warning(8, 2, 3))   # m=48, P(>=1 FP)=91.5%
print(obf_boundaries([0.25, 0.5, 0.75, 1.0]))
```

`family_size_warning` is meant to be run at *design* time, from the pre-registration, before any data exists. Discovering `m = 48` after the fleet has finished is discovering it too late.

## The Failure It Prevents

**Declaring a result that the data does not support, then defending it with a number that was never valid.** The two concrete forms:

- *Breadth*: 48 tests, one comes back at `p = 0.03`, it gets written up. The 91.5% chance of at least one false positive means this outcome is the *expected* one under a completely null intervention. The correction is not pedantry — without it, the trial has no ability to distinguish a real effect from its own size.
- *Depth*: a fleet is monitored daily; on day 9 the *p*-value dips below 0.05 and the run is stopped. The nominal 5% is really ~13% at four looks, and unbounded at continuous monitoring. Worse, stopping at the moment of maximum apparent effect guarantees the reported effect size is inflated — early stopping and the winner's curse are the same phenomenon in the time dimension.

## Decision Procedure

```
1. At DESIGN time, write down the family: every candidate x horizon x metric
   x planned look you could report. Run family_size_warning.

2. Shrink it. Declare ONE primary endpoint (metric + horizon + population).
   Everything else is secondary/descriptive and gets no inferential claim.
   This is worth more than any correction procedure.

3. For the remaining family, choose the control:
   - a wrong call is expensive (admission, safety, headline) -> Holm
   - findings enter a follow-up queue                        -> BH
   - endpoints have a justified importance order             -> hierarchical
                                                                gatekeeping

4. If you will look before the fleet finishes:
   - pre-register the number of looks and the information fractions
   - pick a spending function (O'Brien-Fleming default)
   - add a FUTILITY boundary; it is cheap and saves compute
   - anything not on that schedule is peeking, not an interim analysis

5. Report raw AND adjusted p-values, plus the family size m. A reader who
   sees only adjusted values cannot audit the family declaration.

6. Report all family members, including the nulls. Reporting only the
   survivors makes the correction unverifiable and is survivorship
   (anti-pattern-catalogue.md, AP-06).

7. If you selected the thing you are testing, also correct the ESTIMATE
   (selection-bias-and-best-of-K.md). Error-rate control does not fix
   effect-size inflation; they are separate corrections.
```

## RED Scenario

> A results notebook prints a 48-row table of `candidate × horizon × metric` with a `p` column, sorted ascending. The top row reads `p = 0.007`. The summary says: *"Candidate C4 significantly improves downstream accuracy at the 5k horizon (p = 0.007)."* A Slack thread notes the fleet was stopped two days early "once the signal was clear".

**The catch:** both inflations at once. The family is 48 tests, where P(≥1 false positive under the global null) = 91.5% — a single `p = 0.007` among 48 is unremarkable. And the fleet was stopped on the basis of the result, which invalidates the nominal α entirely and inflates the reported effect at the same time. Sorting the table ascending and quoting the top row is the selection step made visible.

**GREEN behaviour:** *"Two corrections needed, and they are independent.*
>
> *Breadth: with m = 48, Holm-adjusted p for the smallest raw p = 0.007 is 0.336 — not significant. BH gives a q-value that may survive if several other rows are also small; report both, along with m and the full 48-row table including nulls, so the family is auditable. The right structural fix is upstream: declare one primary endpoint (one metric, one horizon) before the next fleet, which takes m from 48 to 1 for the decision that matters.*
>
> *Depth: stopping when the signal looked clear means the nominal 0.05 was never the operating error rate, and the reported effect for C4 is inflated by the same mechanism as the winner's curse — you stopped at a local maximum. There is no post-hoc correction that recovers a valid p from an unplanned stop; the honest description is 'exploratory'.*
>
> *What this trial can support: C4 is a **candidate for confirmation**, not a finding. Pre-register C4, one metric, one horizon, a fleet size from [power](power-and-sample-size-for-paired-designs.md), an O'Brien–Fleming look schedule with a futility boundary, and run it on units this trial never touched. If C4 is real it will survive; that run is the one you cite."*

## Cross-References

- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — the complementary correction: error rate vs effect size
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — where the family and look schedule get declared
- [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md) — horizons as a family dimension, and horizon shopping
- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — composite endpoints as a way to shrink the family
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — correction lowers effective α, which raises the `n` you need
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-05 (uncorrected multiple comparisons), AP-07 (horizon shopping), AP-10 (unplanned stopping)
