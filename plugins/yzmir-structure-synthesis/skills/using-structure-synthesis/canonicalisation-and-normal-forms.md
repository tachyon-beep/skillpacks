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

For the hash built on top of the canonical form, see `equivalence-detection-and-semantic-hashing.md`.

**Pipeline order is fixed across this pack, and it is load-bearing**: cheap legality checks (shape/type inference, acyclicity, interface-contract arity) → **canonicalisation** → the full structural-verification gate, *including reachability and dead-node detection*, run on the canonical form. Reachability runs after canonicalisation because dead-node elimination is canonicalisation's job; a raw graph with a legitimately-generated dangling node would fail an every-node-reaches-output check that fired too early. And identity is assigned once, on the canonical form, so a "cleared candidate" hash means the same thing everywhere. See `structural-verification.md` for the gate itself.

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

> **Representation caveat**: `networkx.DiGraph` holds at most one edge per `(u, v)` pair. A grammar whose operators can consume the *same* producer on two different ports needs `MultiDiGraph` (or ports folded into the edge key). The discipline in this sheet is unchanged; the plumbing is not. This is not a footnote about exotic grammars — it constrains what a rewrite is *allowed to do*, because a rewrite that would need a parallel edge cannot be applied on a `DiGraph` without destroying an existing one. See "The Second Splice Trap" below.

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

### The Second Splice Trap: an Edge the Graph Cannot Hold

Proving the operator is an identity is necessary but still not sufficient. The splice rewrites `pred → n → succ` into `pred → succ`, and that is semantics-preserving **only when `pred → succ` is a new edge**. Take the residual-passthrough pattern the identity-op set exists to serve:

```
a[linear] ──────────port 1──────────▶ out[residual_add]
   └─port 0─▶ id[identity] ─port 0─▶ out          # out computes a + a = 2a
```

`id` is degree-(1,1) and its operator is grammar-declared identity, so the splice fires — and `add_edge("a", "out", in_port=0)` does not add anything, it **overwrites** the existing `("a", "out")` edge, because `DiGraph` cannot hold two. Three edges become one, the merge's in-degree drops from 2 to 1, and the canonical form computes `a` instead of `2a`. That is RED Scenario 1's cardinal sin arriving through the front door, with the operator proof satisfied — and it false-merges too, since the corrupted form is now byte-identical to a genuinely single-input `residual_add(a)`.

The guard is one line: **if `pruned.has_edge(pred, succ)` already, do not splice — keep the identity node.** On a `DiGraph` the identity op is only splice-safe when the spliced edge is fresh; the graph's inability to carry the parallel edge is a real constraint on the rewrite, not a plumbing detail to route around. On a `MultiDiGraph` (ports folded into the edge key) the limitation lifts and the splice is unconditionally safe. Keeping the node costs nothing downstream: it is a stable, deterministic normal form either way, and both copies of a relabeled pair keep it.

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

Refine **bidirectionally** — each round folds in both the upstream (signature, port) pairs and the downstream ones. Nodes that differ in what they feed become distinguishable, and the refinement now separates every pair whose difference is visible from either direction.

Bidirectional refinement is **necessary but not sufficient**, and this is the point where most canonicalisers stop one step short. Whatever residue refinement leaves must be resolved by *orbit-aware individualization*, never by raw node IDs — the raw-ID shortcut has a runnable counterexample, and it is in "What the Tie-Break May Decide" below.

## Executable Decision Procedure

The full implementation with both RED scenarios fixed, plus the property tests that keep them fixed. This is the same implementation quoted in `equivalence-detection-and-semantic-hashing.md` — the two sheets must stay in lock-step, because the hash is computed over this function's output:

```python
import hashlib

IDENTITY_OPS = {"identity", "residual_passthrough"}  # grammar-declared, never inferred


class CanonicalisationBudgetExceeded(RuntimeError):
    """Raised, never swallowed. Falling back to a raw-ID tie-break when the
    individualization search runs long would silently restore the false split
    this procedure exists to remove — and a canonicaliser that quietly degrades
    its own output under load is not a canonicaliser."""


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


def _refine(g, signature):
    """BIDIRECTIONAL WL-style refinement to its fixed point — in-edges and
    out-edges both, with ports. Forward-only refinement is RED Scenario 2.
    Stops when the induced partition stops getting finer."""
    for _ in range(len(signature)):
        new_sig = {}
        for n in g.nodes:
            incoming = sorted(
                (signature[p], g.edges[p, n]["in_port"]) for p in g.predecessors(n)
            )
            outgoing = sorted(
                (signature[s], g.edges[n, s]["in_port"]) for s in g.successors(n)
            )
            new_sig[n] = _digest(signature[n], incoming, outgoing)
        if len(set(new_sig.values())) == len(set(signature.values())):
            return new_sig          # partition stable
        signature = new_sig
    return signature


def _labeling_key(g, ranked):
    """Total-order key for ONE candidate labeling, used only to pick the
    lexicographic minimum across individualization branches. Depends on the
    labeling and the graph, never on the raw node names."""
    index = {n: i for i, n in enumerate(ranked)}
    return (
        tuple(g.nodes[n]["op"] for n in ranked),
        tuple(sorted((index[u], index[v], g.edges[u, v]["in_port"]) for u, v in g.edges)),
    )


def _canonical_order(g, signature, budget):
    """Orbit-aware individualization-refinement. Refine; if nodes are still
    tied, individualize each member of the target cell in turn, re-refine, and
    keep the lexicographically smallest labeling any branch produces. Every
    choice made here is isomorphism-invariant — cell selection by (size,
    signature), individualization by SIGNATURE not by node ID — so two
    isomorphic graphs explore the same set of labelings and pick the same
    minimum."""
    signature = _refine(g, signature)
    cells = {}
    for n in g.nodes:
        cells.setdefault(signature[n], []).append(n)
    tied = [c for c in cells.values() if len(c) > 1]
    if not tied:
        return sorted(g.nodes, key=lambda n: signature[n])   # discrete: done

    target = min(tied, key=lambda c: (len(c), signature[c[0]]))  # smallest cell
    best_order = best_key = None
    for v in sorted(target):        # iteration order cannot change the minimum
        budget[0] -= 1
        if budget[0] < 0:
            raise CanonicalisationBudgetExceeded(
                f"individualization search exhausted its budget; largest tied "
                f"cell has {len(target)} members"
            )
        branch = dict(signature)
        branch[v] = _digest("individualized", signature[v])   # NOT _digest(v)
        ranked = _canonical_order(g, branch, budget)
        key = _labeling_key(g, ranked)
        if best_key is None or key < best_key:
            best_order, best_key = ranked, key
    return best_order


def canonicalise(g: nx.DiGraph, outputs, leaf_budget: int = 10_000) -> nx.DiGraph:
    """Semantics-preserving normal form.
    1. dead-node elimination against DECLARED outputs (a disconnected node has
       out-degree zero too, and is not thereby a legitimate output)
    2. identity-node splicing: grammar-declared identity ops only, never by
       degree — and only when the spliced edge is FRESH (see The Second Splice
       Trap; on a DiGraph the rewrite would otherwise overwrite a real edge)
    3. deterministic relabel via bidirectional refinement plus orbit-aware
       individualization; the raw node ID never breaks a tie
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
                and pruned.nodes[n]["op"] in IDENTITY_OPS   # the operator proof
            ):
                pred = next(pruned.predecessors(n))
                succ = next(pruned.successors(n))
                if pruned.has_edge(pred, succ):
                    continue        # the representation proof: DiGraph cannot
                                    # hold the parallel edge, so add_edge would
                                    # OVERWRITE a real one. Keep the identity.
                pruned.add_edge(pred, succ, in_port=pruned.edges[n, succ]["in_port"])
                pruned.remove_node(n)
                changed = True

    base = {n: _digest("op", pruned.nodes[n]["op"]) for n in pruned.nodes}
    ranked = _canonical_order(pruned, base, [leaf_budget])
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

# --- Property 4: depth-1 automorphic branches unify. The merge op must be one
#     the grammar DECLARES commutative (`add`), so two producers on the same
#     port is the canonical port assignment rather than a port-rule violation;
#     `concat` is port-sensitive and would be the wrong fixture here. ---
def test_automorphic_branches_unify():
    ha = make_graph({"in": "linear", "p": "relu", "q": "relu", "out": "add"},
                    [("in", "p", 0), ("in", "q", 0), ("p", "out", 0), ("q", "out", 0)])
    hb = make_graph({"in": "linear", "x": "relu", "y": "relu", "out": "add"},
                    [("in", "x", 0), ("in", "y", 0), ("x", "out", 0), ("y", "out", 0)])
    assert canonical_bytes(canonicalise(ha, {"out"})) == canonical_bytes(canonicalise(hb, {"out"}))

# --- Property 5: the identity splice never collapses a parallel edge. ---
def test_identity_splice_preserves_residual_passthrough():
    # out = residual_add(a via port 1, identity(a) via port 0) -- computes 2a
    g = make_graph({"a": "linear", "id": "identity", "out": "residual_add"},
                   [("a", "out", 1), ("a", "id", 0), ("id", "out", 0)])
    c = canonicalise(g, {"out"})
    merge = next(n for n in c.nodes if c.nodes[n]["op"] == "residual_add")
    assert c.in_degree(merge) == 2, "splice collapsed 2a into a"
    single = make_graph({"a": "linear", "out": "residual_add"}, [("a", "out", 0)])
    assert canonical_bytes(c) != canonical_bytes(canonicalise(single, {"out"}))
    # ...and the splice still fires when the edge it creates is genuinely fresh
    fresh = make_graph({"a": "linear", "id": "identity", "b": "relu", "c": "linear"},
                       [("a", "id", 0), ("id", "b", 0), ("b", "c", 0)])
    assert len(canonicalise(fresh, {"c"}).nodes) == 3

# --- Property 6: depth->=2 parallel branches. The regression that plain
#     refinement + a raw-ID tie-break CANNOT pass. ---
def make_same_port_deep_branch_pair(merge_op="add"):
    """Two parallel TWO-node chains into the same port of a commutative merge,
    with the mid-chain wiring swapped between the copies. The two `relu`s are
    in one orbit and so are the two `sigmoid`s, but resolving those two orbits
    INDEPENDENTLY (which is what a raw-ID tie-break does) picks a pairing that
    is not an automorphism. Isomorphic; must canonicalise identically."""
    nodes = {"in": "linear", "p1": "relu", "p2": "relu",
             "q1": "sigmoid", "q2": "sigmoid", "out": merge_op}
    common = [("in", "p1", 0), ("in", "p2", 0), ("q1", "out", 0), ("q2", "out", 0)]
    d1 = make_graph(nodes, common + [("p1", "q1", 0), ("p2", "q2", 0)])
    d2 = make_graph(nodes, common + [("p1", "q2", 0), ("p2", "q1", 0)])
    return d1, d2

def test_deep_parallel_branches_do_not_split():
    d1, d2 = make_same_port_deep_branch_pair()
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        d1, d2,
        node_match=lambda x, y: x["op"] == y["op"],
        edge_match=lambda x, y: x["in_port"] == y["in_port"],
    )
    assert matcher.is_isomorphic()      # same structure, provably
    assert canonical_bytes(canonicalise(d1, {"out"})) == canonical_bytes(canonicalise(d2, {"out"}))

# --- Property 7: the fix must not buy unification with false merges. A pair
#     with the same op multiset and the same degree sequence, NOT isomorphic. ---
def test_near_miss_still_separates():
    a = make_graph({"in": "linear", "p1": "relu", "p2": "relu",
                    "q1": "sigmoid", "q2": "sigmoid", "out": "add"},
                   [("in", "p1", 0), ("in", "p2", 0), ("p1", "q1", 0),
                    ("p2", "q2", 0), ("q1", "out", 0), ("q2", "out", 0)])
    b = make_graph({"in": "linear", "p1": "relu", "p2": "relu",
                    "q1": "sigmoid", "q2": "sigmoid", "out": "add"},
                   [("in", "p1", 0), ("in", "p2", 0), ("p1", "q1", 0),
                    ("q1", "q2", 0), ("p2", "out", 0), ("q2", "out", 0)])
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        a, b,
        node_match=lambda x, y: x["op"] == y["op"],
        edge_match=lambda x, y: x["in_port"] == y["in_port"],
    )
    assert not matcher.is_isomorphic()  # genuinely different structures
    assert canonical_bytes(canonicalise(a, {"out"})) != canonical_bytes(canonicalise(b, {"out"}))

test_disconnected_node_is_actually_dead()
test_idempotence()
test_order_independence()
test_downstream_distinction_does_not_split()
test_automorphic_branches_unify()
test_identity_splice_preserves_residual_passthrough()
test_deep_parallel_branches_do_not_split()
test_near_miss_still_separates()
print("canonicalisation properties verified")
```

All eight tests are cheap enough to run on every candidate in CI, not just as a one-off unit test. **Idempotence is the single highest-value test a canonicaliser can have** — a canonicaliser that is not idempotent has not found a normal form, it has found a random walk that happens to terminate somewhere. The RED-2 regression test is the second-highest: it is the one that fails when someone "simplifies" the refinement back to one direction. Property 6 is the third, and it is the one that fails when someone "simplifies" the individualization search back to a raw-ID tie-break.

Note what idempotence structurally *cannot* catch: a canonicaliser that false-splits is usually still perfectly idempotent — each of the two wrong forms is a fixed point of its own. Idempotence proves you reached *a* fixed point, never that both copies reached the *same* one. Only adversarial equivalent-pair tests do that.

## What the Tie-Break May Decide

After bidirectional refinement reaches its fixed point, some nodes may still share a signature. **Nothing may resolve that residue by raw node ID.** The tempting shortcut — `sorted(..., key=(signature, node_id))` — is wrong, and the failure is not hypothetical:

Take two parallel two-node chains, `in → relu → sigmoid → merge`, twice, both feeding the same port of a commutative merge, and swap which `relu` feeds which `sigmoid` between the two copies. The graphs are isomorphic (`DiGraphMatcher` with `node_match`/`edge_match` confirms it). Bidirectional refinement can never separate the twin `relu`s — their upstream and downstream signatures are identical at every round, so this is not a matter of running more rounds — and it can never separate the twin `sigmoid`s either. A raw-ID tie-break then resolves those **two orbits independently**, and the combination it picks is a *pairing*, not an automorphism: one copy gets `(relu₀→sigmoid₀, relu₁→sigmoid₁)`, the other `(relu₀→sigmoid₁, relu₁→sigmoid₀)`. Different canonical bytes. Different hashes. A false split, in a structure family — Inception-style multi-op parallel cells — that a NAS grammar emits routinely. Property 6 above is exactly this pair.

The load-bearing distinction the shortcut elides:

- **Independently-swappable twins are tie-break-safe.** Two nodes whose transposition *by itself* is an automorphism (Property 4's depth-1 branches) can be named either way with no observable difference. This is the case the raw-ID shortcut actually handles.
- **Same-orbit is strictly weaker than that.** Nodes can be automorphic — some automorphism maps one to the other — while their bare transposition is *not* a symmetry, because the automorphism that moves them must move other nodes too. Depth-≥2 branches are the smallest example. Conflating "same orbit" with "freely interchangeable" is the whole bug.

**The sufficient construction is orbit-aware individualization-refinement** (the nauty/bliss skeleton, in miniature): refine to the fixed point; if any cell is still tied, individualize each member of the smallest tied cell in turn, re-refine, recurse, and take the lexicographic minimum of the resulting labelings. Every decision in that loop is isomorphism-invariant — cell selection by `(size, signature)`, individualization by *signature* rather than node ID — so two isomorphic graphs explore the same set of candidate labelings and return the same minimum. That makes the procedure a *complete* canonical form for typed port-labeled DAGs, not a heuristic with rare blind spots.

**Completeness is bought with cost, and the price is real.** The search branches over the smallest tied cell at each level, so a grammar that can emit *k* interchangeable parallel branches costs up to O(k!) labelings in the worst case. Measured on the reference implementation, canonicalising *k* parallel two-node chains into one commutative merge:

| k | wall clock | branch expansions |
|---|---|---|
| 4 | ~3 ms | 40 |
| 5 | ~20 ms | 205 |
| 6 | ~0.2 s | 1,236 |
| 7 | ~0.9 s | 8,659 |
| 8 | — | exceeds the default budget; **raises** |

So the shipped `leaf_budget=10_000` admits up to **seven** fully interchangeable parallel branches and deliberately refuses the eighth. That is a real ceiling, not a safety margin: know it before you raise it, because the next step up costs roughly another factor of *k*.

Three mitigations, in order of preference: (1) bound branch multiplicity in the grammar itself (`typed-graph-grammars.md` ceilings) — most NAS cells top out at 4–6 branches, comfortably inside the default; (2) keep the explicit `leaf_budget` and let it **raise**, never silently fall back to a raw-ID tie-break, because a canonicaliser that degrades under load produces exactly the false splits it was built to prevent — a loud stop is a bug report, a quiet downgrade is corrupted archive data; (3) if graphs genuinely get large and symmetric, move to a real automorphism-pruning implementation (nauty, bliss, or a port of their orbit-pruning) rather than reinventing it. Note also what the canonicaliser does *not* do: it never validates acyclicity. `nx.ancestors` and the refinement loop both run on a cyclic graph and return a plausible-looking form, so the cheap legality checks must reject cycles before canonicalisation ever sees the candidate.

The asymmetry of consequences is what justifies paying that cost at all: when the canonical forms of two equivalent graphs *match*, downstream is correct; when they falsely *split*, no later check catches it — the exact-isomorphism fallback in `equivalence-detection-and-semantic-hashing.md` only fires when hashes already agree. False splits are the failure mode you must test for proactively, because nothing downstream will ever surface them for you. Generate the symmetric structures your grammar can actually express — parallel branches at **every depth your grammar permits**, port permutations of commutative merges, repeated subblocks — and assert their relabeled variants unify. Any pair that doesn't is a canonicaliser bug.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "This node is obviously a pass-through, I can see it in the diagram" | Visual resemblance is not proof; the deleted-relu bug looks identical to a real pass-through until you check the operator |
| "Refining from predecessors is enough — a node is what its inputs make it" | Two nodes with identical upstream worlds can play different downstream roles (RED Scenario 2); a mainstream graph library shipped this exact blindness for years |
| "Refinement is bidirectional now, so breaking the leftover ties by node ID is safe" | Only for twins whose bare transposition is itself an automorphism. Same-orbit is weaker than that, and two orbits resolved independently pick a non-automorphism — Property 6's counterexample |
| "The nodes are in the same orbit, so it doesn't matter which one we call `n0`" | It matters as soon as a *second* tied orbit exists: the automorphism that swaps one pair may be forced to move the other pair too, and independent choices break the correspondence |
| "The node computes the identity, so splicing it out is provably safe" | Two proofs are needed, not one: the operator is an identity **and** the spliced edge does not already exist. On a `DiGraph` the second failure overwrites a real edge and changes what the candidate computes |
| "Checking `canon(canon(x)) == canon(x)` is redundant, I already tested it once" | Idempotence must hold for every input class the generator can produce, not once on a hand-picked example; test it as a property |
| "Python's `hash()` is faster than sha256 for signature compression" | Builtin string hashing is randomized per process; the canonical order would differ between runs, and every persisted identity built on it dies with the process that made it |
| "We'll canonicalise for speed, so a little extra pruning is fine" | Extra pruning beyond provable equivalence is optimization, not canonicalisation — it belongs in the compiler/lowering stage, not here |
| "The grammar doesn't have many identity ops, so degree-based pruning is close enough" | "Close enough" silently corrupts exactly the candidates that don't fit the common case — those are the ones worth generating |
| "Sorting the inputs of every add/merge node is obviously safe" | Only for operators the grammar *declares* commutative; `concat` and weighted merges are port-sensitive, and sorting them changes semantics |
| "We can canonicalise the parameters after the graph shape is fixed" | Parameter-layout canonicalisation carries the same proof discipline as structural canonicalisation — a "cleaner" layout that changes values is a different candidate |

## Red Flags Checklist

- [ ] **Pruning rule keyed on graph shape alone** (degree, fan-in/out) rather than a grammar-declared identity-op set
- [ ] **Identity splice applied without checking whether the spliced edge already exists** — on a `DiGraph` that overwrites a real edge and changes the computed function
- [ ] **Signature refinement reads only one direction** — predecessors only (or successors only); the port-asymmetric regression test does not exist
- [ ] **Raw node IDs break ties after refinement** — the tie-break residue must be resolved by orbit-aware individualization, not by the labels the generator happened to emit
- [ ] **The individualization search silently falls back** to a raw-ID tie-break on timeout or budget exhaustion instead of raising
- [ ] **Builtin `hash()` anywhere in the canonical path** — process-randomized, nondeterministic across runs
- [ ] **No idempotence test** in the canonicaliser's test suite
- [ ] **No order-independence test** — canonical form never checked against a permuted-input version of the same graph
- [ ] **No symmetric-structure tests** — the grammar can express parallel branches or commutative merges, but no test unifies their relabeled variants
- [ ] **Symmetric-structure tests exist but only at depth 1** — single-node branches pass under a raw-ID tie-break; multi-node branches on the same port are the case that actually discriminates
- [ ] **Canonicaliser has a "quality" or "efficiency" flag** — any option that changes output beyond a fixed normal form is not canonicalisation
- [ ] **New simplification rules added without a semantics proof** — "this pattern showed up in generated candidates and looked prunable"
- [ ] **Canonical form depends on which library version produced it** — see `equivalence-detection-and-semantic-hashing.md` for why this is fatal downstream

## Diagnostic Questions

1. **What is your identity-op set, and where is it declared?** If the answer is "we infer pass-throughs structurally," you have RED Scenario 1 waiting. Then ask the follow-up: **does the splice check that the edge it creates is new?** If not, the residual-passthrough pattern silently loses an input.
2. **Does your refinement read out-edges?** Open the loop and look. If it folds in predecessors only, construct the port-asymmetric pair from RED Scenario 2 against your own grammar and run it.
3. **What happens if you canonicalise twice?** If nobody has run `canon(canon(x)) == canon(x)` as a property test across generator output, the answer is unknown, which means the answer is no.
4. **What breaks ties after refinement?** If the answer is "the raw node ID," you have the Property-6 false split whether or not anyone has constructed it yet — build the depth-2 same-port pair against your own grammar and run it. The only sufficient answer is orbit-aware individualization with an isomorphism-invariant cell choice.
5. **Which operators are declared commutative, and by whom?** If port-sorting happens for any operator without a written grammar declaration, semantics can change silently.
6. **Is any step's output process-dependent?** `PYTHONHASHSEED`, dict iteration relied on for order, unpinned library internals — run the canonicaliser in two fresh processes and diff the bytes.

## Cross-References

- **The hash built on the canonical form, its version discipline, and both directions of the equivalence claim**: `equivalence-detection-and-semantic-hashing.md`
- **Both halves of the legality gate this sits between** — the cheap checks (shape, type, acyclicity, contract arity) that run *before* canonicalisation, and the full gate including reachability/dead-node detection that runs *after*, on the canonical form: `structural-verification.md`
- **Why diversity must be measured after canonicalisation, not before**: `diversity-and-mode-collapse.md`
- **How archived candidates use canonical identity for dedup**: `lineage-mutation-and-recombination.md`
- **The full anti-pattern catalogue, including "canonicaliser became a generator"**: `synthesis-anti-patterns.md`
