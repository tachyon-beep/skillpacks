---
name: diversity-and-mode-collapse
description: "Use when measuring whether a generated pool is actually diverse - diversity must be measured in canonical or functional space, never raw syntax, and duplicate rate after canonicalisation is the honest metric. Use when a best-of-K pool looks varied but might be one candidate wearing K different serializations."
---

# Diversity and Mode Collapse

## When to Use

- Measuring or reporting how diverse a generated pool actually is
- Debugging "best-of-K looks fine in logs but downstream selection keeps picking similar things"
- Diagnosing suspected mode collapse in a generator's training
- Reviewing a diversity metric before trusting it in a paper, dashboard, or acceptance gate

This sheet is about *measuring* diversity honestly; `generation-strategies.md` covers *producing* it in the first place, and `canonicalisation-and-normal-forms.md` / `equivalence-detection-and-semantic-hashing.md` provide the machinery this sheet's metric depends on.

## Core Principle

**Diversity measured on raw syntax is not measuring diversity — it's measuring how many different ways the generator's serialization happens to write down however many distinct ideas it actually had.** A generator with genuine mode collapse (one real idea, endlessly restated) can look perfectly diverse under a raw metric, because raw graphs differ in node IDs, insertion order, and incidental structure that canonicalisation exists specifically to strip away.

The honest metric is **duplicate rate after canonicalisation**: generate a pool, canonicalise every member, count distinct canonical hashes, and report that count against the pool size. Anything computed before canonicalisation is measuring the wrong object.

## The RED Scenario: Raw-Syntax Diversity Is a Fiction

A generator that emits fresh random node IDs per sample — ordinary, common behavior, not a bug in isolation — makes every raw serialization look different even when the underlying structure never changes:

```python
def make_pool():
    """A best-of-8 pool that is structurally ONE candidate, relabeled 8
    different ways -- exactly what a generator with fresh per-sample node
    IDs produces even when its actual topology choice never varies."""
    pool = []
    base_ops = {"a": "linear", "b": "relu", "c": "linear"}
    base_edges = [("a", "b", 0), ("b", "c", 0)]
    for i in range(8):
        relabel = {"a": f"a{i}", "b": f"b{i}", "c": f"c{i}"}
        ops = {relabel[k]: v for k, v in base_ops.items()}
        edges = [(relabel[s], relabel[d], p) for s, d, p in base_edges]
        pool.append(make_graph(ops, edges))
    return pool

def raw_serialize(g):
    """Naive 'diversity' measure: serialize in whatever order the generator emitted."""
    lines = [f"NODE {n} op={g.nodes[n]['op']}" for n in g.nodes]
    lines += [f"EDGE {u}->{v}" for u, v in g.edges]
    return "\n".join(lines)

pool = make_pool()
raw_diversity = len({raw_serialize(g) for g in pool})
print(f"raw-syntax distinct count: {raw_diversity} / {len(pool)}")  # 8 / 8 — looks perfectly diverse
```

Eight raw serializations, eight distinct strings, a diversity dashboard reading "100% unique." **The generator produced one idea, eight times.** Anything downstream that trusts this number — a reported coverage estimate, a claim that best-of-K=8 gives the judge real options, a decision that mode collapse isn't a concern — is trusting a measurement of relabeling noise.

### The GREEN Fix: Canonicalise, Then Count

```python
canonical_diversity = len({
    semantic_hash(g, outputs={[n for n in g.nodes if g.out_degree(n) == 0][0]})
    for g in pool
})
print(f"canonical distinct count: {canonical_diversity} / {len(pool)}")  # 1 / 8 — the truth

assert raw_diversity == 8
assert canonical_diversity == 1
```

Using the `canonicalise` / `semantic_hash` implementation from `canonicalisation-and-normal-forms.md` and `equivalence-detection-and-semantic-hashing.md`, the same pool reports 1 distinct candidate out of 8. That is the number that should reach a dashboard, a paper, or a decision about whether the generator needs `learning-objectives-for-generators.md`'s collapse remedies.

## Functional Diversity: One Level Further

Canonical diversity catches structural duplication. It does not catch **functional duplication** — two structurally distinct canonical forms that happen to compute the same function (e.g., two different orderings of commuting linear operations, or two topologically different ways of expressing an equivalent computation the canonicaliser wasn't designed to unify). Where functional equivalence can be checked cheaply (e.g., numerically, by comparing outputs across a fixed probe set of inputs), report both numbers: canonical duplicate rate is the cheap, structural floor; functional duplicate rate is the more expensive, more honest ceiling. The gap between them is itself informative — a large gap means the canonicaliser is under-normalizing, which is a bug to take back to `canonicalisation-and-normal-forms.md`, not a diversity problem to paper over here.

## Remedies, in Order of How Directly They Address the Cause

1. **Confirm the latent (or other variation source) actually reaches the decision that determines topology** — see `generation-strategies.md`'s RED scenario for the "best-of-K with no actual K" failure this often turns out to be.
2. **An explicit min-over-K or winner-take-all training objective** — rather than training every sample toward the same target, train only the closest sample per example, which pressures the K samples to spread out and cover different regions of plausible structure. See `learning-objectives-for-generators.md`.
3. **A contrastive term from prior failures/duplicates** — penalize the generator for producing something that canonicalises to a hash already common in its own recent output.
4. **Widen the grammar, cautiously** — sometimes the space genuinely doesn't contain enough distinct useful structures at the current expressiveness level; see `search-space-evolution-and-explosion-control.md` before doing this, since it has its own cost curve.

## Red Flags Checklist

- [ ] **Diversity metric computed before canonicalisation** — on raw graphs, raw hashes, or generator-internal IDs
- [ ] **No duplicate-rate number reported at all** — only pool size, treated as a proxy for diversity
- [ ] **Dashboard or paper reports "100% unique" for a generator with fresh per-sample IDs** — a near-certain sign of the RED scenario
- [ ] **Functional-duplicate rate never measured**, and the gap to canonical-duplicate rate never estimated or flagged
- [ ] **Mode-collapse remedy proposed without first confirming the latent/variation source actually reaches topology decisions**

## Cross-References

- **The canonical form and hash this metric depends on**: `canonicalisation-and-normal-forms.md`, `equivalence-detection-and-semantic-hashing.md`
- **Why the pool might not be varying in the first place**: `generation-strategies.md`
- **Training objectives that directly target diversity**: `learning-objectives-for-generators.md`
- **Whether the grammar itself is the bottleneck**: `search-space-evolution-and-explosion-control.md`
