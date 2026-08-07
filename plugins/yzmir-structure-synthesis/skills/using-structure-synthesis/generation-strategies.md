---
name: generation-strategies
description: Use when choosing how a generator samples candidates - deterministic direct generation, latent-conditioned sampling, stochastic best-of-K pools, or mutation/recombination - and when deciding whether a demonstrated coverage failure actually justifies escalating to a flow or diffusion model rather than fixing the simpler generator.
---

# Generation Strategies

## When to Use

- Choosing how a generator should sample from the grammar for the first time
- A best-of-K pool looks suspiciously uniform (see also `diversity-and-mode-collapse.md` for how to measure this honestly)
- Deciding whether to reach for a flow or diffusion model
- Reviewing whether "K samples" is actually producing K different sources of variation

This sheet is about *how* a generator samples; `typed-graph-grammars.md` bounds *what* it may sample, `graph-representations-for-generation.md` covers *what format* it emits in, and `learning-objectives-for-generators.md` covers what it's trained to optimize.

## Core Principle

**Best-of-K only works if there are K actual sources of difference behind the K samples. Calling a generator K times is not the same as generating K distinct candidates — it is the same as generating K distinct candidates *only if something varies between the calls that the generator actually responds to*.**

This is the most common way a generation strategy silently fails: the code looks like best-of-K (a loop that calls `generate()` K times and collects the results), the logging shows K entries, and the pool is one candidate wearing K different row indices.

## The Escalation Ladder

Start at the bottom. Move up only when the level below has been tried and has a *demonstrated* coverage failure — not a suspicion, not "it seems limited," an actual measured gap between what the task needs and what the simpler generator can produce.

1. **Deterministic direct generation** — one request in, one candidate out, no randomness. Cheapest to train, cheapest to debug, cheapest to verify (one candidate per request means the verifier's load is minimal). Sufficient whenever the request itself, plus diagnostic context, determines a good-enough answer.
2. **Latent-conditioned generation** — an explicit latent variable conditions the output; sampling the latent produces different candidates from the same request. This is where genuine best-of-K sampling starts being possible.
3. **Stochastic best-of-K pools** — K latent draws (or K samples from a stochastic decoding process) per request, with downstream selection choosing among them. Requires the latent conditioning from level 2 to actually be effective — see the RED scenario below for what happens when it isn't.
4. **Retrieved-parent mutation / lineage recombination** — see `lineage-mutation-and-recombination.md`; sampling starts from an archive of prior candidates rather than from scratch.
5. **Flow or diffusion models** — last resort, justified only by a demonstrated coverage failure of levels 1–4 on the actual task distribution.

The ladder exists because verification cost, debuggability, and training cost all increase climbing it, while the actual coverage need for most structure-synthesis tasks is met well below the top. A small deterministic or latent-conditioned generator that covers the grammar's useful region is strictly preferable to a diffusion model that also covers it, because everything downstream — verification, canonicalisation, diversity measurement — is cheaper against a smaller, better-understood generator.

## The RED Scenario: Best-of-K With No Actual K

A generator with no explicit latent, called K times because the code loops K times:

```python
def best_of_k_NO_LATENT(generate_fn, k):
    """Every call is conditioned on nothing but the request itself. A
    deterministic generator called K times returns the same candidate K times."""
    return [generate_fn() for _ in range(k)]

def deterministic_generator():
    return "block(linear,relu,linear)"   # no arguments -- same output every call

pool = best_of_k_NO_LATENT(deterministic_generator, k=8)
print(len(set(pool)))  # 1 -- eight rows, one candidate
```

This is not a contrived mistake. It happens whenever "add stochasticity" is implemented as "call the same deterministic function more times," or when a nominal source of randomness (e.g., unfixed dropout at inference time) doesn't actually reach the parts of the model that determine graph topology. The pool has eight entries and one candidate. Anything downstream that assumes a best-of-K pool contains K independent draws — diversity metrics, coverage estimates, "probability the pool contains a good candidate" — is now silently wrong.

### The GREEN Fix: An Explicit Latent That Actually Conditions Output

```python
import random

def best_of_k_LATENT_CONDITIONED(generate_fn, k, rng):
    return [generate_fn(latent=rng.random()) for _ in range(k)]

def latent_conditioned_generator(latent):
    width = 8 if latent < 0.5 else 16
    return f"block(linear{width},relu,linear{width})"

rng = random.Random(3)
pool2 = best_of_k_LATENT_CONDITIONED(latent_conditioned_generator, k=8, rng=rng)
assert len(set(pool2)) > 1
print(f"latent-conditioned pool: {len(set(pool2))} distinct candidates out of 8")
```

The fix isn't "add randomness somewhere" — it's confirming that the specific value being varied actually reaches a decision point in the generator that changes the output. A latent that's sampled but only fed into an unused pathway is the same bug wearing a different disguise. This is checkable directly: sample the latent across its range and confirm the *canonical form* (not the raw output — see `canonicalisation-and-normal-forms.md`) of the result actually changes.

## Temperature and Diversity Control

For any stochastic generator (levels 2 and up), sampling temperature is the cheapest diversity knob — and the most commonly misread one. Three facts govern its use:

1. **Temperature trades diversity against validity, and the trade must be measured, not assumed.** Raising temperature pushes probability mass toward the distribution's tails — which contain both the novel structures you want and the illegal or incoherent ones you don't. Structural-rejection rate (`structural-verification.md`) rises with temperature. The honest way to set it: sweep temperature, and for each setting plot distinct-canonical-forms-per-K against rejection rate. Pick the operating point from that curve. A temperature chosen because it's the framework default, or because the samples "looked varied," has not been chosen at all.
2. **Temperature-induced diversity must survive canonicalisation to count.** Higher temperature reliably increases *raw* variation. Whether it increases *canonical* diversity is an empirical question with a specific failure mode: if distinct-canonical count barely moves as temperature rises while raw variation soars, the extra entropy is being spent on relabeling, reordering, and other differences the canonicaliser strips — heat without light. Measure with `diversity-and-mode-collapse.md`'s duplicate-rate metric at each temperature setting, never with raw-sample inspection.
3. **Temperature is a serving-time knob, not a repair for a collapsed model.** If the pool is collapsed at *every* temperature — distinct-canonical count stays near 1 across the sweep — the generator's conditional distribution is degenerate, and no amount of sampling entropy will conjure modes the model doesn't have. That is a training-objective problem: see `learning-objectives-for-generators.md` for min-over-K and contrastive remedies.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "We call generate() K times in a loop, so it's best-of-K" | Best-of-K requires K sources of variation the generator responds to, not K function calls |
| "Dropout gives us enough randomness for diversity" | Dropout noise that doesn't reach the topology-determining part of the model produces K samples that canonicalise to the same structure — see `diversity-and-mode-collapse.md` for how to check |
| "Diffusion models are strictly more expressive, we should just use one" | Expressiveness isn't the bottleneck for most structure-synthesis tasks; verification and debugging cost scale with model complexity regardless of whether the extra expressiveness is used |
| "Our deterministic generator seems limited, let's add a latent" | "Seems limited" is a hypothesis; measure the actual coverage gap against real requests before adding the complexity of levels 2+ |
| "Best-of-K with K=32 must be diverse, that's a big pool" | Pool size and diversity are different numbers; see `diversity-and-mode-collapse.md` for the honest metric (duplicate rate after canonicalisation) |
| "We'll just turn up the temperature until the pool is diverse enough" | Temperature buys tail mass, which contains illegal candidates as well as novel ones — and its raw variation may canonicalise away entirely; measure the distinct-canonical vs. rejection curve before trusting the knob |

## Red Flags Checklist

- [ ] **"Best-of-K" implemented as a loop over a deterministic function** with no per-call source of variation
- [ ] **A latent variable exists but was never confirmed to change the canonical output** across its sampled range
- [ ] **Escalated straight to flow/diffusion** without a measured coverage failure of a simpler generator on the actual task
- [ ] **No test comparing pool size to distinct-canonical-form count** — the gap between them is the actual diversity signal
- [ ] **Latent sampled from a distribution that doesn't match what the model was trained on** (e.g., sampling outside the training-time latent range)

## Cross-References

- **The honest way to measure whether a strategy actually produced diverse candidates**: `diversity-and-mode-collapse.md`
- **What conditions the request itself, separate from the sampling strategy**: `conditioning-on-context-and-contracts.md`
- **The training objective that shapes what the generator learns to vary**: `learning-objectives-for-generators.md`
- **Sampling from an archive instead of from scratch**: `lineage-mutation-and-recombination.md`
- **The format the generator emits, which affects what "sampling" can even vary**: `graph-representations-for-generation.md`
