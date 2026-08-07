---
name: canonicalisation-and-normal-forms
description: Use when reducing generated graphs to a stable normal form - dead-node elimination, duplicate/identity removal, deterministic node ordering, canonical parameter layout - and when a canonicaliser needs to prove it never changes what a candidate computes and never splits one structure into two identities.
---

# Canonicalisation and Normal Forms

## When to Use

- Building or reviewing the pass that turns a raw generated graph into a stable, comparable identity
- Debugging "two candidates that should be the same structure keep coming out different" (usually a canonicalisation gap, not a generation bug)
- Adding a new simplification rule to an existing canonicaliser
- Auditing whether a canonicaliser has quietly started optimizing for something other than a stable normal form

For the hash built on top of the canonical form, see `equivalence-detection-and-semantic-hashing.md`. For the legality checks that should run *before* canonicalisation, see `structural-verification.md`.

## Core Principle

**Canonicalisation answers one question: given this graph, what is the one representative form of everything semantically identical to it? That sentence carries two proof obligations, and they fail in opposite directions.** Every rewrite must be semantics-preserving — a canonicaliser that changes what a candidate computes corrupts every downstream identity at once. And every pair of semantically identical graphs must land on the *same* representative — a canonicaliser that lets incidental raw-graph details (node IDs, edge insertion order, which branch the generator happened to emit first) leak into the normal form splits one structure into several identities, silently inflating every diversity and dedup measurement built on top of it.

A canonicaliser is not a simplifier, not an optimizer, and not a linter. Those tools are allowed to change behavior in exchange for some other benefit (speed, readability, size). A canonicaliser is allowed to change *nothing except syntax* — the serialized form differs, the computed function does not.

This matters because everything downstream trusts the canonical form as an identity:

- Equivalence detection and semantic hashing (`equivalence-detection-and-semantic-hashing.md`) assume two structures canonicalise to the same form if and only if they compute the same function.
- Diversity measurement (`diversity-and-mode-collapse.md`) counts distinct canonical forms, not distinct raw graphs.
- Archives and lineages (`lineage-mutation-and-recombination.md`) key on canonical identity to detect "we've generated this exact structure before."

If canonicalisation fails in either direction, every one of those downstream systems is now wrong in a way that is invisible until someone notices a candidate behaves differently than its canonical form implied — or notices that a "diverse" pool keeps converging to one behavior.

## What Canonicalisation Is Allowed to Do

Only transformations provably equivalent under the graph's declared semantics:

| Transformation | Why it's safe |
|---|---|
| **Dead-node elimination** | A node with no path to any declared output cannot affect the result by definition |
| **Duplicate/identity removal** | A node proven to compute the identity function on its only input can be spliced out |
| **Deterministic node/edge ordering** | Serialization order carries no semantic content; fixing it removes a spurious source of difference |
| **Canonical parameter layout** | Reordering equivalent parameter storage (e.g., row vs. column layout for a commutative op) does not change the function computed |
| **Canonical port assignment for commutative operators** | If the grammar declares an operator's inputs order-insensitive (an unweighted add, a max-pool over branches), sorting which producer feeds which port is a pure relabeling — *only* for operators the grammar explicitly declares commutative |
| **Common-subexpression identification** | Two nodes proven to compute the identical function of identical inputs may be merged — this requires *proof*, not resemblance |

Everything on this list requires a proof obligation, not a heuristic. "This node looks like a pass-through" is not a proof. "This node's operator is drawn from the identity-function subset of the grammar, and its parameters are the identity parameterization" is a proof. The same discipline applies to commutativity: `concat` is not commutative, a weighted sum is not commutative in its weights, and an operator is only port-sortable if the grammar says so in writing.

> **Representation caveat**: `networkx.DiGraph` holds at most one edge per `(u, v)` pair. A grammar whose operators can consume the *same* producer on two different ports needs `MultiDiGraph` (or ports folded into the edge key). The discipline in this sheet is unchanged; the plumbing is not.

## Worked Example

A generator emitted a three-node chain — node IDs are whatever the generator happened to use:

```
x1[linear] ──port 0──▶ x2[relu] ──port 0──▶ x3[linear]
```

A correct canonicaliser recognizes:

- Every node reaches the declared output `x3` — nothing is dead.
- `x2`'s operator is `relu`, which is **not** in the grammar's identity-function subset (it computes `max(x, 0)`, not `x`) — it must be kept.
- The node IDs carry no meaning. A deterministic relabel driven by *structure* (operator plus connectivity), not by the raw IDs, assigns `n0, n1, n2` such that any relabeled copy of this chain — `foo/bar/baz`, `b/c/a`, whatever — lands on byte-identical output.

The canonical form is the same three-op chain with fixed labels. Nothing about the computed function changed, and nothing about the raw labels survived.

## RED Scenario 1: Deleting a Real Nonlinearity

A common and dangerous shortcut: treat every node with in-degree 1 and out-degree 1 as a "pass-through" and splice it out, because most such nodes generated in practice *are* redundant relabeling or identity-initialized residual arms. This heuristic is wrong, and it is wrong silently:

```python
# WRONG — assumes every degree-(1,1) node is an identity / pass-through
def canonicalise_WRONG(g):
    g = g.copy()
    for n in list(g.nodes):
        if g.in_degree(n) == 1 and g.out_degree(n) == 1:
            pred = next(g.predecessors(n))
            succ = next(g.successors(n))
            g.add_edge(pred, succ, in_port=g.edges[n, succ]["in_port"])
            g.remove_node(n)   # BUG: deletes the node without checking its op
    return g
```

Run against the worked example, this deletes `x2` (the `relu`) because it has in-degree 1 and out-degree 1 — exactly like a genuine pass-through would. The result is `x1[linear] → x3[linear]`: two linear layers with no nonlinearity between them, collapsible to a single linear map. **The canonicaliser turned a nonlinear block into a linear one.** Any candidate that happened to route through this bug would silently lose its nonlinearity, and nothing downstream — hashing, verification, evaluation — would flag it, because as far as the pipeline is concerned the canonical form *is* the candidate.

The fix is one condition: only splice a degree-(1,1) node whose operator is a member of the grammar's declared identity-function subset (`identity`, a residual-scale-1.0 branch, a birth-time gate proven to compute `x`) — never by degree alone. That condition is folded into the full implementation below, along with a second dead-node trap: a node with no outgoing edges is not necessarily a legitimate output — it might be a fully disconnected node that was never reachable at all. Dead-node elimination must be defined against the candidate's **declared** outputs, not against "any node with out-degree zero."

## RED Scenario 2: Directionally Blind Refinement

The deterministic-relabel step is where the *other* failure direction lives — the false split. A natural implementation refines each node's structural signature from its **predecessors**: start from the operator, repeatedly fold in the (signature, port) pairs of everything upstream, then sort nodes by final signature to assign canonical labels. It reads correctly. It is wrong, and the failure is invisible on chains — it needs a graph where two nodes differ only in their *downstream* role:

```python
import networkx as nx

def make_graph(ops, edges):
    g = nx.DiGraph()
    for node_id, op in ops.items():
        g.add_node(node_id, op=op)
    for src, dst, port in edges:
        g.add_edge(src, dst, in_port=port)
    return g

def canonical_bytes(g):
    """Pinned serialization — never rely on networkx's internal iteration order."""
    lines = [f"NODE {n} op={g.nodes[n]['op']}" for n in sorted(g.nodes)]
    lines += [f"EDGE {u}->{v} port={g.edges[u, v]['in_port']}" for u, v in sorted(g.edges)]
    return "\n".join(lines).encode("utf-8")

def canonicalise_FORWARD_ONLY(g):
    """WRONG — signature refinement reads predecessors only. Nodes that differ
    only in their DOWNSTREAM role keep identical signatures forever, and the
    node-ID tie-break then freezes the generator's arbitrary raw labels into
    the 'canonical' form."""
    order = list(nx.topological_sort(g))
    signature = {n: g.nodes[n]["op"] for n in order}
    for _ in range(len(order) + 1):
        signature = {
            n: (signature[n], tuple(sorted(
                (signature[p], g.edges[p, n]["in_port"]) for p in g.predecessors(n)
            )))
            for n in order
        }
    ranked = sorted(order, key=lambda n: (signature[n], n))
    relabel = {old: f"n{i}" for i, old in enumerate(ranked)}
    return nx.relabel_nodes(g, relabel, copy=True)

# Two relu branches, distinguished ONLY by which in_port of the merge they feed.
# g1 and g2 are the same structure — the isomorphism swaps a and b.
g1 = make_graph(
    {"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
    [("in", "a", 0), ("in", "b", 0), ("a", "out", 0), ("b", "out", 1)],
)
g2 = make_graph(
    {"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
    [("in", "a", 0), ("in", "b", 0), ("a", "out", 1), ("b", "out", 0)],
)

matcher = nx.algorithms.isomorphism.DiGraphMatcher(
    g1, g2,
    node_match=lambda x, y: x["op"] == y["op"],
    edge_match=lambda x, y: x["in_port"] == y["in_port"],
)
assert matcher.is_isomorphic()   # same structure, provably

b1 = canonical_bytes(canonicalise_FORWARD_ONLY(g1))
b2 = canonical_bytes(canonicalise_FORWARD_ONLY(g2))
assert b1 != b2                  # ...and yet: two different "canonical" forms
print("forward-only refinement split one structure into two identities")
```

Both `relu` nodes see the same upstream world — `linear`, port 0 — so predecessor-only refinement can never tell them apart, and the tie-break falls through to the raw node IDs. One structure, two canonical forms, two hashes: a false split. Every downstream consumer now counts this candidate twice in diversity metrics and misses it once in archive dedup.

If this failure mode sounds familiar: it is the same in-edges/out-edges blindness networkx patched in its own graph-hash routine in v3.5, with a versioned-output warning attached — the exact library warning quoted in `equivalence-detection-and-semantic-hashing.md`. The bug class is common enough that a mainstream library shipped it for years. Assume your hand-rolled refinement has it until a port-asymmetric test proves otherwise.

### The GREEN Fix

Refine **bidirectionally** — each round folds in both the upstream (signature, port) pairs and the downstream ones. Nodes that differ in what they feed become distinguishable, and the tie-break only ever has to decide between nodes the refinement genuinely cannot separate (see "What the Tie-Break May Decide" below for why that residue is safe).

## Executable Decision Procedure

The full implementation with both RED scenarios fixed, plus the property tests that keep them fixed. This is the same implementation quoted in `equivalence-detection-and-semantic-hashing.md` — the two sheets must stay in lock-step, because the hash is computed over this function's output:

```python
import hashlib

IDENTITY_OPS = {"identity", "residual_passthrough"}  # grammar-declared, never inferred

def _digest(*parts) -> str:
    """Deterministic signature compression. NEVER Python's builtin hash() here:
    string hashing is randomized per process (PYTHONHASHSEED), which would make
    the canonical ordering differ between runs — a nondeterministic canonicaliser
    is a contradiction in terms. sha256 also keeps per-round signatures at
    constant size instead of nesting tuples exponentially on high-fan-in graphs."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def canonicalise(g: nx.DiGraph, outputs) -> nx.DiGraph:
    """Semantics-preserving normal form.
    1. dead-node elimination against DECLARED outputs (a disconnected node has
       out-degree zero too, and is not thereby a legitimate output)
    2. identity-node splicing: grammar-declared identity ops only, never by degree
    3. deterministic relabel via BIDIRECTIONAL WL-style refinement — in-edges and
       out-edges both, with ports; forward-only refinement is RED Scenario 2
    """
    outputs = set(outputs)
    assert outputs <= set(g.nodes), "declared outputs must exist in the graph"
    live = set(outputs)
    for o in outputs:
        live |= nx.ancestors(g, o)
    pruned = g.subgraph(live).copy()

    changed = True
    while changed:
        changed = False
        for n in list(pruned.nodes):
            if (
                n not in outputs
                and pruned.in_degree(n) == 1
                and pruned.out_degree(n) == 1
                and pruned.nodes[n]["op"] in IDENTITY_OPS   # the proof obligation
            ):
                pred = next(pruned.predecessors(n))
                succ = next(pruned.successors(n))
                pruned.add_edge(pred, succ, in_port=pruned.edges[n, succ]["in_port"])
                pruned.remove_node(n)
                changed = True

    order = list(nx.topological_sort(pruned))
    signature = {n: _digest("op", pruned.nodes[n]["op"]) for n in order}
    for _ in range(len(order) + 1):  # n rounds reach the refinement fixed point
        new_sig = {}
        for n in order:
            incoming = sorted(
                (signature[p], pruned.edges[p, n]["in_port"])
                for p in pruned.predecessors(n)
            )
            outgoing = sorted(
                (signature[s], pruned.edges[n, s]["in_port"])
                for s in pruned.successors(n)
            )
            new_sig[n] = _digest(signature[n], incoming, outgoing)
        signature = new_sig

    ranked = sorted(order, key=lambda n: (signature[n], n))
    relabel = {old: f"n{i}" for i, old in enumerate(ranked)}
    return nx.relabel_nodes(pruned, relabel, copy=True)


# --- Property 0: a disconnected node is dead, not a phantom output ---
def test_disconnected_node_is_actually_dead():
    g = make_graph(
        {"a": "linear", "b": "relu", "c": "linear", "dangling": "linear"},
        [("a", "b", 0), ("b", "c", 0)],
    )
    assert len(canonicalise(g, outputs={"c"}).nodes) == 3

# --- Property 1: idempotence. canon(canon(x)) must equal canon(x). ---
def test_idempotence():
    g = make_graph(
        {"a": "linear", "id": "identity", "b": "relu", "c": "linear"},
        [("a", "id", 0), ("id", "b", 0), ("b", "c", 0)],
    )
    c1 = canonicalise(g, outputs={"c"})
    c2 = canonicalise(c1, outputs={n for n in c1.nodes if c1.out_degree(n) == 0})
    assert canonical_bytes(c1) == canonical_bytes(c2), "not idempotent"
    assert len(c1.nodes) == 3                                   # identity spliced
    assert any(c1.nodes[n]["op"] == "relu" for n in c1.nodes)   # relu kept

# --- Property 2: order-independence. Insertion/edge order must not matter. ---
def test_order_independence():
    ga = make_graph({"a": "linear", "b": "relu", "c": "linear"}, [("a", "b", 0), ("b", "c", 0)])
    gb = make_graph({"c": "linear", "a": "linear", "b": "relu"}, [("b", "c", 0), ("a", "b", 0)])
    assert canonical_bytes(canonicalise(ga, {"c"})) == canonical_bytes(canonicalise(gb, {"c"}))

# --- Property 3: RED-2 regression. The port-asymmetric pair must unify. ---
def test_downstream_distinction_does_not_split():
    assert canonical_bytes(canonicalise(g1, {"out"})) == canonical_bytes(canonicalise(g2, {"out"}))

# --- Property 4: truly automorphic branches are tie-break-safe. ---
def test_automorphic_branches_unify():
    ha = make_graph({"in": "linear", "p": "relu", "q": "relu", "out": "concat"},
                    [("in", "p", 0), ("in", "q", 0), ("p", "out", 0), ("q", "out", 0)])
    hb = make_graph({"in": "linear", "x": "relu", "y": "relu", "out": "concat"},
                    [("in", "x", 0), ("in", "y", 0), ("x", "out", 0), ("y", "out", 0)])
    assert canonical_bytes(canonicalise(ha, {"out"})) == canonical_bytes(canonicalise(hb, {"out"}))

test_disconnected_node_is_actually_dead()
test_idempotence()
test_order_independence()
test_downstream_distinction_does_not_split()
test_automorphic_branches_unify()
print("canonicalisation properties verified")
```

All five tests are cheap enough to run on every candidate in CI, not just as a one-off unit test. **Idempotence is the single highest-value test a canonicaliser can have** — a canonicaliser that is not idempotent has not found a normal form, it has found a random walk that happens to terminate somewhere. The RED-2 regression test is the second-highest: it is the one that fails when someone "simplifies" the refinement back to one direction.

## What the Tie-Break May Decide

After bidirectional refinement reaches its fixed point, some nodes may still share a signature. The final `sorted(..., key=(signature, node_id))` breaks those ties by raw node ID — which looks like exactly the raw-label leakage RED Scenario 2 forbids. The difference is *what kind* of nodes can still be tied:

- If two nodes share a fixed-point bidirectional signature because they are **genuinely automorphic** — swapping them is a symmetry of the graph — then either tie-break choice produces byte-identical canonical output (Property 4 above tests this). The raw ID chose *which name* each twin gets, and the twins are interchangeable, so nothing observable depends on the choice.
- WL-style refinement is not a complete isomorphism decider in general — there exist graph families it cannot fully distinguish. On small typed DAGs over a port-labeled grammar these blind spots are rare, but "rare" is a property to *measure against your grammar*, not assume: generate the symmetric structures your grammar can actually express (parallel branches, port permutations of commutative merges, repeated subblocks) and test that relabeled variants unify. Any pair that doesn't is a canonicaliser bug of exactly the RED-2 class.
- The asymmetry of consequences matters: when the canonical forms of two equivalent graphs *match*, downstream is correct; when they falsely *split*, no later check catches it — the exact-isomorphism fallback in `equivalence-detection-and-semantic-hashing.md` only fires when hashes agree. False splits are the failure mode you must test for proactively, because nothing downstream will ever surface them for you.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "This node is obviously a pass-through, I can see it in the diagram" | Visual resemblance is not proof; the deleted-relu bug looks identical to a real pass-through until you check the operator |
| "Refining from predecessors is enough — a node is what its inputs make it" | Two nodes with identical upstream worlds can play different downstream roles (RED Scenario 2); a mainstream graph library shipped this exact blindness for years |
| "Checking `canon(canon(x)) == canon(x)` is redundant, I already tested it once" | Idempotence must hold for every input class the generator can produce, not once on a hand-picked example; test it as a property |
| "Python's `hash()` is faster than sha256 for signature compression" | Builtin string hashing is randomized per process; the canonical order would differ between runs, and every persisted identity built on it dies with the process that made it |
| "We'll canonicalise for speed, so a little extra pruning is fine" | Extra pruning beyond provable equivalence is optimization, not canonicalisation — it belongs in the compiler/lowering stage, not here |
| "The grammar doesn't have many identity ops, so degree-based pruning is close enough" | "Close enough" silently corrupts exactly the candidates that don't fit the common case — those are the ones worth generating |
| "Sorting the inputs of every add/merge node is obviously safe" | Only for operators the grammar *declares* commutative; `concat` and weighted merges are port-sensitive, and sorting them changes semantics |
| "We can canonicalise the parameters after the graph shape is fixed" | Parameter-layout canonicalisation carries the same proof discipline as structural canonicalisation — a "cleaner" layout that changes values is a different candidate |

## Red Flags Checklist

- [ ] **Pruning rule keyed on graph shape alone** (degree, fan-in/out) rather than a grammar-declared identity-op set
- [ ] **Signature refinement reads only one direction** — predecessors only (or successors only); the port-asymmetric regression test does not exist
- [ ] **Builtin `hash()` anywhere in the canonical path** — process-randomized, nondeterministic across runs
- [ ] **No idempotence test** in the canonicaliser's test suite
- [ ] **No order-independence test** — canonical form never checked against a permuted-input version of the same graph
- [ ] **No symmetric-structure tests** — the grammar can express parallel branches or commutative merges, but no test unifies their relabeled variants
- [ ] **Canonicaliser has a "quality" or "efficiency" flag** — any option that changes output beyond a fixed normal form is not canonicalisation
- [ ] **New simplification rules added without a semantics proof** — "this pattern showed up in generated candidates and looked prunable"
- [ ] **Canonical form depends on which library version produced it** — see `equivalence-detection-and-semantic-hashing.md` for why this is fatal downstream

## Diagnostic Questions

1. **What is your identity-op set, and where is it declared?** If the answer is "we infer pass-throughs structurally," you have RED Scenario 1 waiting.
2. **Does your refinement read out-edges?** Open the loop and look. If it folds in predecessors only, construct the port-asymmetric pair from RED Scenario 2 against your own grammar and run it.
3. **What happens if you canonicalise twice?** If nobody has run `canon(canon(x)) == canon(x)` as a property test across generator output, the answer is unknown, which means the answer is no.
4. **What breaks ties after refinement?** If raw node IDs break ties between nodes that are *not* provably automorphic, raw labels are leaking into the canonical form.
5. **Which operators are declared commutative, and by whom?** If port-sorting happens for any operator without a written grammar declaration, semantics can change silently.
6. **Is any step's output process-dependent?** `PYTHONHASHSEED`, dict iteration relied on for order, unpinned library internals — run the canonicaliser in two fresh processes and diff the bytes.

## Cross-References

- **The hash built on the canonical form, its version discipline, and both directions of the equivalence claim**: `equivalence-detection-and-semantic-hashing.md`
- **The legality checks that should already have run before canonicalisation** (shape, cycles, contracts): `structural-verification.md`
- **Why diversity must be measured after canonicalisation, not before**: `diversity-and-mode-collapse.md`
- **How archived candidates use canonical identity for dedup**: `lineage-mutation-and-recombination.md`
- **The full anti-pattern catalogue, including "canonicaliser became a generator"**: `synthesis-anti-patterns.md`
