---
name: learning-objectives-for-generators
description: "Use when choosing the training objective for a structure generator - reconstruction, functional-effect matching, min-over-K/ranking, contrastive learning from failures - and when a generator's own auxiliary utility head is at risk of filtering its own output pool, which is the generator grading its own examination."
---

# Learning Objectives for Generators

## When to Use

- Choosing what loss to train a structure generator against
- Reviewing whether a generator's auxiliary prediction head has started influencing which candidates it returns
- Deciding how to use failed/rejected candidates as training signal rather than discarding them
- Debugging "the pool the judge sees never contains the candidates that turn out best"

This sheet is about training the generator; `generation-strategies.md` covers how it samples once trained, and `diversity-and-mode-collapse.md` covers measuring whether training produced real variation.

## Core Principle

**Distinguish an unranked proposal API from a utility-guided search API.** When the contract promises an unfiltered proposal pool for independent downstream evaluation, silently applying an auxiliary utility threshold violates that contract. A declared search policy may legitimately filter or rank by predicted utility; report the proposal budget, selection rule and withheld candidates as relevant, and evaluate the complete selected pipeline on held-out tasks. Predicted utility cannot substitute for structural legality or independent evidence of quality.

## Objective Families

| Objective | What it trains | Where it's appropriate |
|---|---|---|
| **Canonical parameter reconstruction** | Reproduce a target candidate's canonical form exactly | Warm-starting a generator from an archive of known-good candidates |
| **Functional-effect matching** | Match a target's measured behavior (not its exact structure) | When multiple structures achieve the same effect and any of them is acceptable |
| **Min-over-K / winner-take-all** | Only the closest of K samples to a target is trained toward it; the rest are left free | Directly counters mode collapse — see `diversity-and-mode-collapse.md` — by not forcing every sample toward the same point |
| **Ranking / pairwise preference** | Relative ordering between two candidates whose true utility is known | When absolute utility is hard to calibrate but relative comparisons are reliable |
| **Contrastive from failures** | Pull generation away from structures a downstream judge has rejected | Using the full history — including failures — as training signal (see `lineage-mutation-and-recombination.md` on archiving failures, not just winners) |

Training objectives do not by themselves define the sampling contract. Choose and document whether evaluation receives unranked proposals or the output of a declared search/selection policy.

## The RED Scenario: An Auxiliary Head That Filters Its Own Pool

This example assumes the API promises the full proposal pool. An auxiliary head silently removes candidates before independent evaluation:

```python
from dataclasses import dataclass

@dataclass
class Candidate:
    name: str
    aux_predicted_utility: float   # the generator's OWN auxiliary head's guess
    true_utility: float            # ground truth; only the downstream judge sees this

def generate_pool_SELF_FILTERED(pool, threshold=0.5):
    """The generator drops candidates its own auxiliary head scored low,
    before the pool ever reaches the downstream judge."""
    return [c for c in pool if c.aux_predicted_utility >= threshold]

def downstream_judge_pick_best(pool):
    """Stand-in for the real evaluator this pack does not implement --
    picks by TRUE utility, which the generator's aux head does not have."""
    return max(pool, key=lambda c: c.true_utility) if pool else None
```

The aux head is a biased, incomplete proxy — as every auxiliary head trained on limited data is, to some degree. Here it badly underrates the candidate that is actually best:

```python
pool = [
    Candidate("A", aux_predicted_utility=0.8, true_utility=0.6),
    Candidate("B", aux_predicted_utility=0.3, true_utility=0.9),  # aux head is wrong about this one
    Candidate("C", aux_predicted_utility=0.6, true_utility=0.5),
]

filtered_pool = generate_pool_SELF_FILTERED(pool, threshold=0.5)
best_from_filtered = downstream_judge_pick_best(filtered_pool)
print([c.name for c in filtered_pool], "-> judge picks:", best_from_filtered.name if best_from_filtered else None)
# ['A', 'C'] -> judge picks: A
```

Candidate B — the true best — never reaches the judge. Not because it was evaluated and lost; because the generator's own imperfect proxy for utility decided it wasn't worth showing anyone. **The self-filtering is invisible from the judge's side**: the judge sees a pool, picks the best member, and has no way to know a better one was silently withheld upstream.

### The GREEN Fix: Report Everything, Attach the Score as Metadata

```python
def generate_pool_HONEST(pool):
    """The full pool is returned; the aux score travels as metadata for the
    downstream judge to use or ignore, never as a filter."""
    return list(pool)

honest_pool = generate_pool_HONEST(pool)
best_from_honest = downstream_judge_pick_best(honest_pool)
print([c.name for c in honest_pool], "-> judge picks:", best_from_honest.name)
# ['A', 'B', 'C'] -> judge picks: B

assert best_from_filtered.name != "B"
assert best_from_honest.name == "B"
```

For an unranked-pool API, return the requested proposals and attach auxiliary scores as metadata. For a search API, an alternative repair is to expose the filtering contract and evaluate its recall, cost and held-out utility against an unfiltered baseline; returning everything is not universally optimal.

## Checks for an Unranked Proposal Contract

- **Training-time only**: shapes gradients during generator training, never called at serving/sampling time to decide pool membership.
- **Reported, not applied**: if computed at serving time at all (e.g., for logging or research), it is attached to the candidate record and passed downstream unfiltered — the receiving system decides what to do with it, and that system is not this pack's concern (it's a downstream evaluator's job — the same boundary drawn in `structural-verification.md`).
- **Retrained on the judge's actual verdicts, not on its own prior predictions**: an aux head that bootstraps from its own past outputs rather than ground truth will confidently reinforce its own biases, including the exact bias that caused the RED scenario above.

## Contract-Violation Checks for an Unranked Proposal API

- [ ] **A generator function's return value is filtered by any internally-computed utility, quality, or confidence score**
- [ ] **An auxiliary head trained on the generator's own past predictions** rather than ground-truth judge verdicts
- [ ] **"Efficiency" or "pre-screening" cited as the reason a pool is smaller than requested**
- [ ] **No test verifying that pool size returned equals pool size requested**, regardless of internal scores
- [ ] **The judge's interface has no way to request "show me everything, unfiltered"** — meaning the option to bypass silent filtering doesn't even structurally exist

## Cross-References

- **Why the pool a generator returns needs to actually vary**: `generation-strategies.md`
- **The honest way to measure whether that variation happened**: `diversity-and-mode-collapse.md`
- **Using failed/rejected candidates as training material without survivorship bias**: `lineage-mutation-and-recombination.md`
- **The parallel invariant on the verification side**: `structural-verification.md`
- **The full anti-pattern catalogue**: `synthesis-anti-patterns.md`
