---
name: lineage-mutation-and-recombination
description: Use when generating candidates by mutating or recombining an archive of prior structures rather than from scratch - parent selection, grammar-respecting mutation operators, interface-compatible crossover, and provenance tracking. Use when a mutation operator risks producing an illegal graph (a cycle, a broken contract) that must still pass the full verification gate.
---

# Lineage, Mutation, and Recombination

## When to Use

- Building a generator that mutates or recombines candidates from an archive rather than generating from scratch
- Designing parent-selection strategy for an evolutionary or archive-driven search
- Reviewing whether a mutation operator can produce an illegal candidate
- Deciding what to keep in the archive — and specifically whether failures belong there

This sheet is one *source* of candidates among several in `generation-strategies.md`'s escalation ladder. Everything a mutation or crossover operator produces is still a raw, untrusted candidate — it still goes through `structural-verification.md` exactly like a from-scratch candidate does.

## Core Principle

**Retrieval and mutation supply candidate material. They are never an approval path.** A candidate that started from a previously-accepted parent is not thereby pre-verified — the mutation operator itself can introduce illegality, and "it came from a good parent" is not evidence about the mutated offspring's legality or quality. Every mutated or recombined candidate re-enters the pipeline at the same point a from-scratch candidate would: raw, unverified, and subject to the full checklist in `structural-verification.md`.

## Parent Selection and the Archive

An archive of prior candidates, keyed by canonical identity (`equivalence-detection-and-semantic-hashing.md`), is the source material for mutation and recombination. Two archive design decisions matter more than the mutation operators themselves:

1. **Retain failures and rejections, not just winners.** An archive that only stores accepted candidates is survivorship bias applied to search: every future mutation starts from a narrower, self-reinforcing slice of the space, and the information in "this family of structures was tried and rejected" — valuable precisely because it prevents re-discovering the same dead end — is thrown away. See `synthesis-anti-patterns.md` for this pattern's full writeup.
2. **Key retrieval on canonical identity, not raw candidate ID.** Two raw candidates that canonicalise to the same structure should count as one entry for diversity accounting and parent-selection purposes — otherwise the archive inherits the exact raw-syntax-diversity illusion described in `diversity-and-mode-collapse.md`.

## The RED Scenario: A Mutation Operator That Breaks the Graph It's Given

A mutation that adds an edge between two randomly chosen nodes, with no check on whether the result is still a DAG:

```python
import networkx as nx

def mutate_ADD_RANDOM_EDGE_WRONG(g, rng):
    """Picks any two distinct nodes and adds an edge between them.
    Does not check whether this creates a cycle."""
    g2 = g.copy()
    src, dst = rng.sample(list(g2.nodes), 2)
    g2.add_edge(src, dst, in_port=0)
    return g2

def make_chain():
    g = nx.DiGraph()
    for n, op in [("a", "linear"), ("b", "relu"), ("c", "linear")]:
        g.add_node(n, op=op)
    g.add_edge("a", "b", in_port=0)
    g.add_edge("b", "c", in_port=0)
    return g
```

A perfectly ordinary parent — a 3-node chain — mutated by adding a single edge from its last node back to its first:

```python
class ForcedBackEdge:
    """Deterministic stand-in for rng.sample, always proposing the
    back-edge (c -> a), to make the failure reproducible here."""
    def sample(self, population, k):
        return ["c", "a"]

g = make_chain()
mutated_wrong = mutate_ADD_RANDOM_EDGE_WRONG(g, ForcedBackEdge())
print(nx.is_directed_acyclic_graph(mutated_wrong))  # False — the mutation created a cycle
```

`c -> a` closes a loop through the existing `a -> b -> c` path. The parent was a perfectly legal, previously-verified candidate. The mutation operator, applied naively, produced something `structural-verification.md`'s cycle check would (correctly) reject — which is exactly why that check has to run on every mutated candidate, not just on from-scratch ones.

### The GREEN Fix: Grammar-Respecting Mutation, Plus Verification Anyway

Constrain the mutation operator to only add edges consistent with a topological order, so it cannot produce a cycle by construction:

```python
def mutate_ADD_EDGE_RESPECTING_ORDER(g, rng):
    """Only adds an edge from an earlier node (in topological order) to a
    later one -- structurally cannot create a cycle in a DAG."""
    g2 = g.copy()
    order = list(nx.topological_sort(g2))
    i, j = sorted(rng.sample(range(len(order)), 2))
    src, dst = order[i], order[j]
    if not g2.has_edge(src, dst):
        g2.add_edge(src, dst, in_port=0)
    return g2

import random
mutated_green = mutate_ADD_EDGE_RESPECTING_ORDER(g, random.Random(1))
assert nx.is_directed_acyclic_graph(mutated_green) is True
print("order-respecting mutation:", nx.is_directed_acyclic_graph(mutated_green))
```

This fix reduces how often mutation produces illegal candidates. **It does not replace the verifier.** Order-respecting edge addition prevents cycles specifically; it says nothing about parameter budgets, shape contracts, or forbidden operations a different mutation operator (say, one that swaps an operator at a node) might violate. Every mutation operator needs its own analysis of what it can break, and the verifier needs to run regardless of that analysis being right.

## Crossover Across Interface-Compatible Subgraphs

Recombination — splicing a subgraph from one archived candidate into another — is legal source material exactly when the spliced-in subgraph's boundary satisfies the same interface contract (`conditioning-on-context-and-contracts.md`) the target position requires: matching input/output shapes and arity at the splice points. A crossover operator that ignores boundary compatibility produces a candidate `structural-verification.md`'s shape-inference check will catch — better to check compatibility before spawning the candidate (cheaper than discovering the mismatch after generation), but the post-hoc check remains mandatory regardless, for the same reason constrained decoding doesn't replace verification (`validity-by-construction-vs-post-hoc.md`).

## Provenance: Recorded, Never Consulted by the Gates

Every mutated or recombined candidate should carry its parent lineage — which archive entries it derived from, which operator produced it — for audit, debugging, and archive-quality analysis. This provenance must never be read by `structural-verification.md`'s legality decision or by any utility judgement: "this candidate's parent was previously accepted" is not evidence the offspring is legal, and treating it as such is the retrieval-as-approval-path failure this sheet is about. Provenance is retained for humans and for training signal (`learning-objectives-for-generators.md`'s contrastive-from-failures objective, for instance) — never for gating.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "The parent was verified, so a small mutation of it should be fine" | Verification is a property of the exact candidate checked, not something that survives an edit by proximity |
| "We only keep the winning candidates in the archive, why store the losers" | Discarding rejected lineages is survivorship bias in the search process itself — future mutation will re-explore and re-reject the same dead ends |
| "Crossover between two legal candidates must produce something legal" | Legal parents with incompatible splice boundaries produce an illegal child; legality doesn't compose across an edit that wasn't checked |
| "Provenance is just for logging, it doesn't affect any decision" | If the field exists on a candidate object passed into a decision function, it's one refactor away from being read there — see `structural-verification.md`'s parallel warning |
| "This mutation operator is simple, it can't introduce a cycle" | The RED scenario's operator is about as simple as a mutation operator gets, and it introduces a cycle on the very first back-edge it happens to pick |

## Red Flags Checklist

- [ ] **A mutated or recombined candidate skips any part of `structural-verification.md`'s checklist** because it derived from a verified parent
- [ ] **Archive stores only accepted candidates** — no record of rejected or failed lineages
- [ ] **Mutation operator has no argument for why it cannot produce an illegal graph**, or that argument has never been tested
- [ ] **Crossover splice points never checked for interface-contract compatibility** before or after the splice
- [ ] **Provenance fields present in the same function signature as a legality or utility decision**

## Cross-References

- **The full legality gate every mutated/recombined candidate must still pass**: `structural-verification.md`
- **The canonical identity the archive keys retrieval on**: `equivalence-detection-and-semantic-hashing.md`
- **Why raw-syntax diversity in the archive is misleading**: `diversity-and-mode-collapse.md`
- **The interface contract crossover splice points must respect**: `conditioning-on-context-and-contracts.md`
- **Using retained failures as training signal**: `learning-objectives-for-generators.md`
- **The full anti-pattern catalogue, including "archive stores only winners"**: `synthesis-anti-patterns.md`
