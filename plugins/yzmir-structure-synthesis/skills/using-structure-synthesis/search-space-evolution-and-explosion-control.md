---
name: search-space-evolution-and-explosion-control
description: "Use when deciding whether to grow a grammar's operator whitelist or topology freedom - gates, ceilings, and the verification-cost curve - and when recognising that search and verification are becoming intractable before that becomes the way you find out."
---

# Search Space Evolution and Explosion Control

## When to Use

- Considering adding operators, topology freedom, or a new staged expressiveness level to an existing grammar
- Verification is getting slower, or the structural-rejection rate is climbing, after a recent grammar change
- Deciding whether "just a few more operators" is actually a small change
- Setting up the reliability-gate criteria a grammar expansion must clear

This sheet governs *when and how much* to grow the grammar defined in `typed-graph-grammars.md`. It assumes that sheet's ceilings and staged-levels model already exists.

## Core Principle

**Grammar growth is combinatorial in verification cost; grammar additions are usually proposed as if they were linear.** "One more operator" sounds like a small, additive change. Against a fixed node and edge ceiling, it multiplies the number of possible per-node operator assignments — and every one of those candidates, reachable or not, is something the verifier, the canonicaliser, and the diversity metric now have to be correct against. The failure this produces is not a crash; it's the search and verification pipeline getting slower and less reliable release over release, with no single change anyone can point to as "the" cause, because no single change was ever tested against the cumulative cost of the ones before it.

## The RED Scenario: "A Few More Operators" Is Not a Small Change

A crude but honest lower-bound estimate of the reachable space at a fixed node/edge ceiling, as a function of whitelist size:

```python
def search_space_size(n_ops: int, n_nodes: int, n_edge_slots: int) -> int:
    """Independent op choice per node times independent presence/absence
    choice per possible edge slot. Deliberately crude -- the point is the
    growth curve, not an exact count."""
    return (n_ops ** n_nodes) * (2 ** n_edge_slots)

N_NODES = 6
N_EDGE_SLOTS = N_NODES * (N_NODES - 1) // 2  # upper-triangle slots for a DAG over a fixed order

sizes = {n_ops: search_space_size(n_ops, N_NODES, N_EDGE_SLOTS) for n_ops in [4, 6, 8, 10]}
for n_ops, size in sizes.items():
    print(f"{n_ops} ops -> search space ~{size:.3e}")

growth_4_to_6 = sizes[6] / sizes[4]
growth_4_to_10 = sizes[10] / sizes[4]
print(f"growth factor 4->6 ops: {growth_4_to_6:.1f}x")
print(f"growth factor 4->10 ops: {growth_4_to_10:.1f}x")

assert growth_4_to_6 > 5      # "a couple more ops" is already a large multiplier
assert growth_4_to_10 > 100   # "not that many more ops" is a two-order-of-magnitude blowup
```

Going from 4 operators to 6 — a change that reads as minor in a PR description — grows the reachable space over 11x at a fixed 6-node ceiling. Going from 4 to 10 grows it over 240x. **None of the ceilings changed.** The node cap, the edge cap, the parameter budget — all exactly as declared in `typed-graph-grammars.md`. Only the whitelist grew, and that alone was enough to make the space the generator, verifier, and canonicaliser all have to handle correctly two orders of magnitude larger.

### The GREEN Fix: A Reliability Gate Before Every Expansion

Growth is not forbidden — it's staged behind a measured health check on the *current* grammar level, so expansion only happens once the tooling has proven it can keep up:

```python
def ready_to_expand(rejection_rate: float, duplicate_rate: float, verify_p95_seconds: float,
                     max_rejection=0.15, max_duplicate=0.30, max_verify_seconds=0.5) -> bool:
    """The CURRENT level must already be healthy on all three axes before
    the grammar is allowed to grow to the next level or gain new operators."""
    return (
        rejection_rate <= max_rejection
        and duplicate_rate <= max_duplicate
        and verify_p95_seconds <= max_verify_seconds
    )

healthy = ready_to_expand(rejection_rate=0.08, duplicate_rate=0.20, verify_p95_seconds=0.12)
unhealthy = ready_to_expand(rejection_rate=0.35, duplicate_rate=0.20, verify_p95_seconds=0.12)
assert healthy is True
assert unhealthy is False
print(f"healthy level ready to expand: {healthy} | unhealthy level ready to expand: {unhealthy}")
```

A grammar with a 35% structural-rejection rate is not ready for more operators — it's a sign the generator is already struggling to reliably produce legal candidates in the *current*, smaller space. Adding operators to a struggling generator does not fix the struggle; it gives the generator more ways to fail in the same proportion, against a verifier that now has more to check per candidate.

## What the Gate Should Measure

| Metric | Why it matters before expanding |
|---|---|
| **Structural-rejection rate** (`structural-verification.md`) | High rejection means the current generator/grammar pairing isn't reliable yet; more expressiveness compounds the problem |
| **Duplicate rate after canonicalisation** (`diversity-and-mode-collapse.md`) | If the current level is already mode-collapsed, expansion won't fix that — fix the collapse first, then decide if expansion is still wanted |
| **Verification p95 latency** | Directly measures the cost curve this sheet is about; a level whose verification time is already climbing will only get worse |
| **Canonicalisation idempotence/stability** (`canonicalisation-and-normal-forms.md`) | A canonicaliser with known bugs at the current level should not be trusted with more structural variety to canonicalise |

All four should be measured, not assumed. A grammar can look healthy on rejection rate while already mode-collapsed, or fast on verification while quietly non-idempotent on canonicalisation — the gate needs all of them, not whichever one is easiest to compute.

## Recognizing Explosion Before It's a Production Incident

Warning signs, roughly in the order they tend to appear:

1. Verification p95 latency creeps up release over release, with no single release identified as the cause.
2. Structural-rejection rate rises even though the generator's training hasn't changed — the space it's sampling into got harder to hit legally.
3. Canonicalisation starts taking a noticeably larger share of per-candidate wall-clock time (more nodes/edges to normalize, more identity-op cases to consider).
4. New operators get added faster than the reliability-gate metrics are reviewed — the gate exists on paper but nobody is checking it before merging a whitelist change.

Any one of these, caught early, is a config change (tighten a ceiling, pause expansion, fix a canonicalisation bug). Caught late, after several ungated expansions have compounded, it's a re-architecture of the verification pipeline under production pressure.

## Red Flags Checklist

- [ ] **A grammar expansion merged with no reliability-gate check against the current level's metrics**
- [ ] **No verification-latency trend tracked over grammar/whitelist changes**
- [ ] **Rejection rate rising and attributed to "the generator needs more training"** without checking whether the grammar just grew
- [ ] **Duplicate rate never checked before deciding the generator needs more expressiveness** (a mode-collapsed generator doesn't need a bigger space, it needs `diversity-and-mode-collapse.md`'s remedies)
- [ ] **Whitelist changes reviewed as isolated PRs** with no visibility into the cumulative combinatorial effect of the last several changes

## Cross-References

- **The grammar and ceilings this sheet governs the growth of**: `typed-graph-grammars.md`
- **The rejection-rate and cost metrics this gate consumes**: `structural-verification.md`
- **The duplicate-rate metric this gate consumes**: `diversity-and-mode-collapse.md`
- **The canonicaliser whose stability this gate depends on**: `canonicalisation-and-normal-forms.md`
- **The full anti-pattern catalogue, including "grammar expanded without a reliability gate"**: `synthesis-anti-patterns.md`
