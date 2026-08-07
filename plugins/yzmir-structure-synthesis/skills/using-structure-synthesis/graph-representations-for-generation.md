---
name: graph-representations-for-generation
description: Use when choosing the encoding a generator actually emits - op sequences, edge lists, adjacency, or a latent code plus decoder - and when a representation choice silently loses information a downstream stage needs, such as branching structure a sequence-only encoding can't express.
---

# Graph Representations for Generation

## When to Use

- Choosing what format a generator's output head actually produces
- Debugging "the generated graph looks wrong after decoding, but the model's loss was fine"
- Reviewing whether a representation can express every structure the grammar allows
- Deciding between op sequences, edge lists, adjacency matrices, or a latent-code-plus-decoder scheme

This sheet is about the *format*, not the *sampling strategy* (see `generation-strategies.md`) or the *legality rules* (see `typed-graph-grammars.md` and `structural-verification.md`). A representation can be perfectly round-trip-faithful and still generate illegal graphs; those are separate concerns.

## Core Principle

**A representation is only as good as its round trip. If encoding a graph and then decoding the result does not reconstruct the same graph, the representation has a blind spot — and a generator trained to emit that representation has been trained to be blind to exactly the structures the encoding can't express, whether or not the grammar allows them.**

This is easy to miss because the blind spot doesn't show up as a training bug. Loss goes down, samples look reasonable, and the representation quietly excludes an entire class of legal structures — usually anything requiring more context than the encoding carries per token.

## Common Representations and What They Cost

| Representation | What it's good at | What it struggles with |
|---|---|---|
| **Op sequence, implicit connectivity** (each token connects to the previous) | Simple, cheap, easy to train autoregressively | Cannot express branching (a node with more than one parent) or skip connections at all |
| **Op sequence with explicit pointers** (each token names its parent indices) | Expresses arbitrary DAG topology | Requires the model to predict valid back-references; harder to keep index-consistent |
| **Edge list** | Directly matches the graph structure; easy to validate | Node ordering/count must be separately specified or inferred |
| **Adjacency matrix** | Fixed-size, easy tensor representation for small graphs | Scales quadratically with node count; typically needs a fixed maximum node count |
| **Latent code → decoder** | Compact, supports smooth interpolation and sampling diversity | Round-trip fidelity depends entirely on decoder quality; requires separately verifying the decoder is faithful |

None of these is universally correct. The right choice depends on what the grammar needs to express (see `typed-graph-grammars.md`) and what the generation strategy needs to sample from (see `generation-strategies.md`).

## The RED Scenario: Implicit Connectivity Can't Express a Residual Block

An op-sequence representation where each token is just an operator name, and connectivity is assumed to be "each node connects to the previous one":

```python
import networkx as nx

def encode_SEQUENCE_ONLY(g, order):
    return [g.nodes[n]["op"] for n in order]

def decode_SEQUENCE_ONLY(tokens):
    g = nx.DiGraph()
    prev = None
    for i, op in enumerate(tokens):
        g.add_node(i, op=op)
        if prev is not None:
            g.add_edge(prev, i, in_port=0)
        prev = i
    return g
```

Encode a genuine residual block — an input that feeds *both* a transform and the final add, so the add node has two parents — and decode it back:

```python
g = nx.DiGraph()
for n, op in [("in", "identity"), ("lin", "linear"), ("add", "residual_add")]:
    g.add_node(n, op=op)
g.add_edge("in", "lin", in_port=0)
g.add_edge("lin", "add", in_port=0)
g.add_edge("in", "add", in_port=1)   # 'add' has TWO parents: 'lin' and 'in' — this is the branch

order = ["in", "lin", "add"]
decoded = decode_SEQUENCE_ONLY(encode_SEQUENCE_ONLY(g, order))
print(g.number_of_edges(), decoded.number_of_edges())  # 3, 2 — the branch edge is gone
```

The original graph has 3 edges. The round-tripped graph has 2. **The skip connection — the entire point of a residual block — cannot be represented at all**, not because the grammar forbids it but because the encoding has no field for "this node has a second parent." A generator trained to emit this representation will never produce a residual structure, regardless of how much training data contained one, because the representation itself cannot carry that information through.

### The GREEN Fix: Make Parent References Explicit

```python
def encode_POINTER_BASED(g, order):
    """Each parent reference is an (index, in_port) PAIR, not a bare index.
    Dropping the port is the same class of bug one level down: the edge
    survives the round trip, the wiring does not."""
    index_of = {n: i for i, n in enumerate(order)}
    return [
        (g.nodes[n]["op"],
         tuple(sorted((index_of[p], g.edges[p, n]["in_port"]) for p in g.predecessors(n))))
        for n in order
    ]

def decode_POINTER_BASED(tokens):
    g = nx.DiGraph()
    for i, (op, _) in enumerate(tokens):
        g.add_node(i, op=op)
    for i, (op, parents) in enumerate(tokens):
        for parent_index, in_port in parents:
            g.add_edge(parent_index, i, in_port=in_port)
    return g

decoded2 = decode_POINTER_BASED(encode_POINTER_BASED(g, order))
assert decoded2.number_of_edges() == g.number_of_edges() == 3
assert (sorted((u, v, d["in_port"]) for u, v, d in decoded2.edges(data=True))
        == [(0, 1, 0), (0, 2, 1), (1, 2, 0)])   # ports survive, not just edges
print("pointer-based round trip preserves all 3 edges AND their port assignments")
```

**Carry the port, not just the parent index.** An encoding that stores `tuple(sorted(index_of[p] for p in g.predecessors(n)))` and decodes with `in_port=0` on every edge round-trips *edge count* faithfully and *wiring* incorrectly — for a portless grammar nothing is lost, but the moment the grammar contains a port-sensitive operator (`concat`, a weighted merge, an attention block with distinct query/key inputs) the decoder silently rewires every candidate onto port 0. Because the edge count still matches, the obvious round-trip assertion passes. If the grammar is genuinely portless, dropping the port is a legitimate simplification — but write that assumption down as a stated limitation next to the encoder, because it is exactly the kind of assumption a later grammar extension invalidates without touching this file. (A grammar that can feed the *same* producer into two ports of one operator needs `MultiDiGraph` on top of this; see the representation caveat in `canonicalisation-and-normal-forms.md`.)

The fix costs something: the generator must now predict valid parent indices *and* valid ports (which the implicit scheme never had to do), and invalid or forward-referencing pointers become a new failure mode `structural-verification.md`'s reachability and cycle checks need to catch. That's a real cost — and it's the cost of being able to express the structures the grammar actually allows, rather than a silent subset of them.

## Choosing a Representation: Ask What the Grammar Needs First

Work from `typed-graph-grammars.md` outward, not the other way around:

1. Does the grammar's topology require branching (multiple parents) or only chains? If only chains, sequence-only is sufficient and cheaper to train.
2. Is node count bounded and small? Adjacency matrices become viable and simple; at larger bounds they waste tensor capacity on mostly-zero entries.
3. Does the generation strategy need smooth interpolation between candidates (for mutation, see `lineage-mutation-and-recombination.md`, or exploration)? A latent-code representation supports that; discrete token sequences do not, without extra machinery.
4. Whatever is chosen, **write the round-trip test before training anything.** It is cheap, and it is the only thing that catches a representational blind spot before a generator has spent a training run learning to live inside it.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "The model's loss is going down, the representation must be fine" | A representation with a blind spot trains a generator that never produces the excluded structures — and loss reflects what the representation *can* express, not what the grammar allows |
| "We'll add branching support later if we need it" | Adding it later means re-encoding the entire training corpus and retraining, not a config flag |
| "Adjacency matrices are simple, let's just use those" | Simple at small node counts; the ceiling from `typed-graph-grammars.md` determines whether that holds — check the node ceiling before committing |
| "A latent-code decoder is more flexible, so it's strictly better" | Flexibility shifts the round-trip-fidelity burden onto the decoder network, which now needs its own faithfulness testing — it isn't free, it's relocated |
| "We tested round-trip on a few examples and it worked" | Test round-trip on the structures that stress the representation specifically — branching, the deepest allowed graph, the widest allowed graph — not just typical examples |

## Red Flags Checklist

- [ ] **No round-trip test exists** for the chosen representation against the grammar's full topology space
- [ ] **Implicit connectivity used for a grammar that allows branching** (multiple parents, skip connections)
- [ ] **A latent-code decoder's faithfulness has never been separately measured** — round-trip fidelity assumed rather than verified
- [ ] **Representation chosen before the grammar's topology requirements were nailed down**
- [ ] **Fixed-size adjacency representation used with no check that it covers the grammar's node ceiling**

## Cross-References

- **The grammar whose topology this representation must be able to express**: `typed-graph-grammars.md`
- **How the generator samples using this representation**: `generation-strategies.md`
- **The checks a decoded, possibly-malformed candidate must pass regardless of representation**: `structural-verification.md`
- **Why representation choice affects measured diversity**: `diversity-and-mode-collapse.md`
