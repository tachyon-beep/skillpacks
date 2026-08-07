---
name: equivalence-detection-and-semantic-hashing
description: Use when detecting whether two generated structures compute the same function, assigning equivalence-class identity, or designing a semantic hash - and when that hash needs to survive serialization-order changes, library upgrades, and canonicaliser revisions rather than merely working on today's test graphs.
---

# Equivalence Detection and Semantic Hashing

## When to Use

- Deciding whether two generated candidates are "the same structure" for deduplication, archiving, or diversity measurement
- Designing or reviewing a semantic hash used as a candidate's stable identity
- Debugging "candidates that should be identical hash differently" or, worse, "candidates that are actually different hash the same"
- A dependency upgrade (a graph library, a serialization format) — or a change to your own canonicaliser — silently changed hash values

For the canonical normal form the hash is built on, see `canonicalisation-and-normal-forms.md` — read that sheet first; this one assumes a correct canonicaliser exists. For where equivalence detection fits in the legality gate, see `structural-verification.md`.

## Core Principle

**A semantic hash is a claim: equal semantics implies equal hash, and distinct semantics implies distinct hash. Both directions are testable, and a hash that has only ever been checked in one direction has only been half-verified.** Most hash bugs that reach production passed the direction someone thought to test and failed the direction nobody did.

The two directions fail differently:

- **Equal semantics, unequal hash** (a false split) quietly inflates measured diversity — a best-of-K pool of one real candidate that hashes to three different identities looks three times more diverse than it is. See `diversity-and-mode-collapse.md`.
- **Distinct semantics, equal hash** (a false merge) quietly conflates two different candidates into one archive entry, silently discarding one of them, or dismisses a genuinely novel candidate as a duplicate of something already tried.

Neither failure crashes anything. Both corrupt every downstream measurement that trusts the hash as identity, and by the time someone notices, the corrupted measurements have already shaped which candidates got kept, mutated, or reported as "the same idea, tried three times."

The two directions also have **asymmetric detectability**, and the architecture must respect it: a false merge can be caught downstream by an exact isomorphism check run when hashes agree; a false split can never be caught downstream, because no later stage compares candidates whose hashes already differ. Test coverage has to be proactive on the false-split side — that is where silent corruption lives.

## The RED Scenarios: Three Traps, All Real and All Reproducible

### Trap 1: bare graph isomorphism ignores labels

`networkx.is_isomorphic()` on plain graphs checks **topology only**. Two graphs with identical structure but completely different operators at every node compare as isomorphic:

```python
import networkx as nx

g1 = nx.DiGraph()
g1.add_node("x1", op="linear"); g1.add_node("x2", op="relu"); g1.add_node("x3", op="linear")
g1.add_edge("x1", "x2"); g1.add_edge("x2", "x3")

g2 = nx.DiGraph()
g2.add_node("x1", op="conv"); g2.add_node("x2", op="sigmoid"); g2.add_node("x3", op="attention")
g2.add_edge("x1", "x2"); g2.add_edge("x2", "x3")

print(nx.is_isomorphic(g1, g2))  # True — same shape, totally different operators
```

Run this and it prints `True`. A verifier or deduplicator built on the bare call will merge a linear-relu-linear block with a conv-sigmoid-attention block, because as far as `is_isomorphic` is concerned they are the same graph. **The fix is `node_match` (and `edge_match` for typed ports), not a different function:**

```python
matcher = nx.algorithms.isomorphism.DiGraphMatcher(
    g1, g2,
    node_match=lambda a, b: a["op"] == b["op"],
    edge_match=lambda a, b: a.get("in_port") == b.get("in_port"),
)
print(matcher.is_isomorphic())  # False — correctly rejects
```

A sheet, command, or codebase that ships bare `is_isomorphic()` as an equivalence check has shipped its own RED scenario as GREEN. Grep for it; there should be zero uses without a `node_match` argument anywhere near candidate structures.

### Trap 2: library hash functions are attribute-blind by default and change across versions

`networkx.weisfeiler_lehman_graph_hash` is a fast, well-known graph hash — and it is dangerous to use directly as a semantic hash for two independent reasons, both reproducible today:

```python
import networkx as nx

g1 = nx.DiGraph(); g1.add_node("a", op="linear"); g1.add_node("b", op="relu"); g1.add_edge("a", "b")
g2 = nx.DiGraph(); g2.add_node("a", op="conv"); g2.add_node("b", op="attention"); g2.add_edge("a", "b")

h1 = nx.weisfeiler_lehman_graph_hash(g1)
h2 = nx.weisfeiler_lehman_graph_hash(g2)
print(h1 == h2)  # True by default — the call ignores node attributes unless told not to

h1b = nx.weisfeiler_lehman_graph_hash(g1, node_attr="op")
h2b = nx.weisfeiler_lehman_graph_hash(g2, node_attr="op")
print(h1b != h2b)  # True — only correct once node_attr is passed explicitly
```

Without `node_attr="op"`, two graphs with completely different operators hash identically — the same failure mode as Trap 1, one layer down. And even with `node_attr` supplied, networkx 3.5 shipped this warning on the same call:

> `UserWarning: The hashes produced for directed graphs changed in version v3.5 due to a bugfix to track in and out edges separately (see documentation).`
>
> `UserWarning: The hashes produced for graphs without node or edge attributes changed in v3.5 due to a bugfix (see documentation).`

That is a library vendor telling you, in the vendor's own words, that upgrading the library silently changes hash values for graphs you have already hashed and archived. **Any pipeline that persists `weisfeiler_lehman_graph_hash` output as a long-lived identity, across a library upgrade, has silently invalidated its own archive without an error, a warning in the log it actually reads, or a way to tell which old hashes are still trustworthy.**

### Trap 3: your own refinement can carry the same bug the library patched

Read that v3.5 warning again: the bugfix was to *track in and out edges separately*. A hand-rolled canonicaliser whose signature refinement reads predecessors only has precisely the blindness networkx patched — two nodes distinguished solely by their downstream role (which port of a merge they feed) never get separated, and isomorphic graphs canonicalise to different forms. That is a **false split**, the direction no downstream check can catch. The full failing example, runnable, is RED Scenario 2 in `canonicalisation-and-normal-forms.md`.

This pack's own reference implementation is a worked example of the consequence, and it has now paid the price **twice**:

- **`structhash-v1`** — predecessor-only refinement. False-split on port-asymmetric branch-and-merge graphs.
- **`structhash-v2`** — bidirectional refinement, raw node ID breaking whatever ties were left. Fixed v1's failure and shipped a subtler one: refinement is necessary but not sufficient, and a raw-ID tie-break resolves two tied orbits independently, which false-splits parallel **multi-node** branches feeding the same port of a commutative merge (Inception-style cells). See RED Scenario 2's "What the Tie-Break May Decide" in `canonicalisation-and-normal-forms.md`.
- **`structhash-v3`** — bidirectional refinement plus orbit-aware individualization-refinement, lexicographic-minimum over branches. No raw label reaches the canonical form at any point.

Each change altered canonical bytes for structures the previous version got wrong, so v1, v2 and v3 hashes are mutually incomparable — and the version field is what makes that incomparability *visible* instead of silent. Note the shape of the v2 lesson specifically: a fix that closes the counterexample you have is not the same as a fix that closes the *class*. A version bump on canonicaliser change is not bureaucracy; it is the only thing standing between "we fixed the canonicaliser" and "we corrupted the archive and called it a fix."

## The GREEN Fix: Own the Serialization, Version the Hash, Never Trust Equality Alone

Three disciplines, applied together:

1. **Canonicalise first** — the hash is computed over the *canonical form* (`canonicalisation-and-normal-forms.md`), never the raw graph. Two semantically identical raw graphs canonicalise to byte-identical serializations; the hash is deterministic from there.
2. **Pin your own serialization** — do not delegate byte layout to a library's internal iteration order or a general-purpose hash function's attribute handling. Serialize nodes and edges yourself, in a fixed, sorted order, with an explicit format.
3. **Version the hash** — a `hash_version` string is part of the hash input. When the canonicalisation algorithm, the serialization format, or a pinned dependency changes in a way that could change output, bump the version. Old archives keep their old hash and are recomputed under the new version rather than silently compared across an undeclared change.

An external graph hash function (Weisfeiler-Leman or otherwise) is legitimate as a **fast pre-filter** — cheap to compute, used to skip the expensive exact check for the overwhelming majority of non-matching pairs — but it is never the authority. Two candidates with equal fast-hash are still confirmed or refuted by the canonical-bytes hash (or, if you don't trust that either, by the exact labeled isomorphism check). Two candidates with unequal fast-hash are correctly assumed non-equivalent without further work, *provided* the fast hash's false-negative rate has actually been measured, not assumed to be zero.

## Executable Decision Procedure

The same `canonicalise` / `canonical_bytes` implementation from `canonicalisation-and-normal-forms.md` — bidirectional refinement, orbit-aware individualization, guarded identity splice, sha256 signature compression — extended with the versioned hash and an equivalence check that tests both directions. **The two sheets must stay byte-for-byte in lock-step**; a drift between them is a hash-version incident:

```python
import hashlib
import networkx as nx

HASH_VERSION = "structhash-v3"  # v1 = predecessor-only refinement (Trap 3);
                                # v2 = bidirectional refinement + raw-ID tie-break
                                #      (still false-split on deep same-port branches);
                                # v3 = orbit-aware individualization, no raw label
                                #      anywhere. Bump on ANY canonicalisation or
                                #      serialization change.
IDENTITY_OPS = {"identity", "residual_passthrough"}

def make_graph(ops, edges):
    g = nx.DiGraph()
    for node_id, op in ops.items():
        g.add_node(node_id, op=op)
    for src, dst, port in edges:
        g.add_edge(src, dst, in_port=port)
    return g

def _digest(*parts) -> str:
    # sha256, never builtin hash(): string hashing is PYTHONHASHSEED-randomized
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

class CanonicalisationBudgetExceeded(RuntimeError):
    """Raised, never swallowed — a silent fallback to a raw-ID tie-break would
    restore exactly the v2 false split."""

def _refine(g, signature):
    """Bidirectional WL refinement to its fixed point (Trap 3's fix)."""
    for _ in range(len(signature)):
        new_sig = {}
        for n in g.nodes:
            incoming = sorted((signature[p], g.edges[p, n]["in_port"])
                              for p in g.predecessors(n))
            outgoing = sorted((signature[s], g.edges[n, s]["in_port"])
                              for s in g.successors(n))
            new_sig[n] = _digest(signature[n], incoming, outgoing)
        if len(set(new_sig.values())) == len(set(signature.values())):
            return new_sig
        signature = new_sig
    return signature

def _labeling_key(g, ranked):
    index = {n: i for i, n in enumerate(ranked)}
    return (tuple(g.nodes[n]["op"] for n in ranked),
            tuple(sorted((index[u], index[v], g.edges[u, v]["in_port"]) for u, v in g.edges)))

def _canonical_order(g, signature, budget):
    """Refine; while ties remain, individualize each member of the smallest
    tied cell, re-refine, and keep the lexicographically smallest labeling.
    Cell choice and individualization are both signature-driven, never
    node-ID-driven — that is what makes the result isomorphism-invariant."""
    signature = _refine(g, signature)
    cells = {}
    for n in g.nodes:
        cells.setdefault(signature[n], []).append(n)
    tied = [c for c in cells.values() if len(c) > 1]
    if not tied:
        return sorted(g.nodes, key=lambda n: signature[n])
    target = min(tied, key=lambda c: (len(c), signature[c[0]]))
    best_order = best_key = None
    for v in sorted(target):
        budget[0] -= 1
        if budget[0] < 0:
            raise CanonicalisationBudgetExceeded(
                f"individualization search exhausted its budget; largest tied "
                f"cell has {len(target)} members")
        branch = dict(signature)
        branch[v] = _digest("individualized", signature[v])   # NOT _digest(v)
        ranked = _canonical_order(g, branch, budget)
        key = _labeling_key(g, ranked)
        if best_key is None or key < best_key:
            best_order, best_key = ranked, key
    return best_order

def canonicalise(g, outputs, leaf_budget=10_000):
    outputs = set(outputs)
    live = set(outputs)
    for o in outputs:
        live |= nx.ancestors(g, o)
    pruned = g.subgraph(live).copy()
    changed = True
    while changed:
        changed = False
        for n in list(pruned.nodes):
            if (n not in outputs and pruned.in_degree(n) == 1 and pruned.out_degree(n) == 1
                    and pruned.nodes[n]["op"] in IDENTITY_OPS):
                pred, succ = next(pruned.predecessors(n)), next(pruned.successors(n))
                if pruned.has_edge(pred, succ):
                    continue   # DiGraph cannot hold the parallel edge: splicing
                               # would OVERWRITE a real one and change semantics
                pruned.add_edge(pred, succ, in_port=pruned.edges[n, succ]["in_port"])
                pruned.remove_node(n)
                changed = True
    base = {n: _digest("op", pruned.nodes[n]["op"]) for n in pruned.nodes}
    ranked = _canonical_order(pruned, base, [leaf_budget])
    relabel = {old: f"n{i}" for i, old in enumerate(ranked)}
    return nx.relabel_nodes(pruned, relabel, copy=True)

def canonical_bytes(g):
    lines = [f"NODE {n} op={g.nodes[n]['op']}" for n in sorted(g.nodes)]
    lines += [f"EDGE {u}->{v} port={g.edges[u, v]['in_port']}" for u, v in sorted(g.edges)]
    return "\n".join(lines).encode("utf-8")

def semantic_hash(g, outputs) -> str:
    """The authoritative identity: HASH_VERSION-scoped digest of our own pinned
    serialization of the canonical form. Never delegate this to a library's
    default hash function — see the three traps above."""
    canon = canonicalise(g, outputs)
    payload = HASH_VERSION.encode() + b"\n" + canonical_bytes(canon)
    return hashlib.sha256(payload).hexdigest()

def _labeled_isomorphic(g1, g2):
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        g1, g2,
        node_match=lambda a, b: a["op"] == b["op"],
        edge_match=lambda a, b: a["in_port"] == b["in_port"],
    )
    return matcher.is_isomorphic()

def graphs_equivalent(g1, out1, g2, out2) -> bool:
    """Hash as a fast filter; exact labeled isomorphism as the authority.
    A hash mismatch is trusted as proof of non-equivalence — which is exactly
    why false splits are the dangerous direction and get proactive tests.
    A hash match is NOT trusted by itself — confirm with the exact check."""
    if semantic_hash(g1, out1) != semantic_hash(g2, out2):
        return False
    return _labeled_isomorphic(canonicalise(g1, out1), canonicalise(g2, out2))


# --- Direction 1: equal semantics -> equal hash ---
def test_equivalent_graphs_hash_equal():
    g1 = make_graph({"x1": "linear", "x2": "relu", "x3": "linear"}, [("x1", "x2", 0), ("x2", "x3", 0)])
    g2 = make_graph({"foo": "linear", "bar": "relu", "baz": "linear"}, [("foo", "bar", 0), ("bar", "baz", 0)])
    assert semantic_hash(g1, {"x3"}) == semantic_hash(g2, {"baz"})
    assert graphs_equivalent(g1, {"x3"}, g2, {"baz"})

# --- Direction 1, adversarial: port-asymmetric branches (v1 regression) ---
def test_branch_and_merge_does_not_falsely_split():
    g1 = make_graph({"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
                    [("in", "a", 0), ("in", "b", 0), ("a", "out", 0), ("b", "out", 1)])
    g2 = make_graph({"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
                    [("in", "a", 0), ("in", "b", 0), ("a", "out", 1), ("b", "out", 0)])
    assert semantic_hash(g1, {"out"}) == semantic_hash(g2, {"out"})
    assert graphs_equivalent(g1, {"out"}, g2, {"out"})

# --- Direction 1, adversarial: DEEP same-port branches (v2 regression).
#     v1's fixture above passes under v2; this one does not. Depth is the
#     discriminating variable, not port asymmetry. ---
def make_same_port_deep_branch_pair(merge_op="add"):
    """Two parallel TWO-node chains into the SAME port of a commutative merge,
    with the mid-chain wiring swapped between the copies. The relus are one
    tied orbit and the sigmoids another; resolving the two orbits independently
    (what a raw-ID tie-break does) picks a pairing that is not an automorphism."""
    nodes = {"in": "linear", "p1": "relu", "p2": "relu",
             "q1": "sigmoid", "q2": "sigmoid", "out": merge_op}
    common = [("in", "p1", 0), ("in", "p2", 0), ("q1", "out", 0), ("q2", "out", 0)]
    d1 = make_graph(nodes, common + [("p1", "q1", 0), ("p2", "q2", 0)])
    d2 = make_graph(nodes, common + [("p1", "q2", 0), ("p2", "q1", 0)])
    return d1, d2

def test_deep_same_port_branches_do_not_falsely_split():
    for merge_op in ("add", "residual_add", "concat"):
        d1, d2 = make_same_port_deep_branch_pair(merge_op)
        assert _labeled_isomorphic(d1, d2), "fixture must be isomorphic to be a fixture"
        assert semantic_hash(d1, {"out"}) == semantic_hash(d2, {"out"})
        assert graphs_equivalent(d1, {"out"}, d2, {"out"})

# --- Direction 1, adversarial: the identity splice must not corrupt semantics ---
def test_residual_passthrough_survives_canonicalisation():
    # out = residual_add(a via port 1, identity(a) via port 0) -- computes 2a
    two_a = make_graph({"a": "linear", "id": "identity", "out": "residual_add"},
                       [("a", "out", 1), ("a", "id", 0), ("id", "out", 0)])
    one_a = make_graph({"a": "linear", "out": "residual_add"}, [("a", "out", 0)])
    canon = canonicalise(two_a, {"out"})
    merge = next(n for n in canon.nodes if canon.nodes[n]["op"] == "residual_add")
    assert canon.in_degree(merge) == 2, "splice collapsed 2a into a"
    assert semantic_hash(two_a, {"out"}) != semantic_hash(one_a, {"out"})   # no false merge
    assert not graphs_equivalent(two_a, {"out"}, one_a, {"out"})

# --- Direction 2: distinct semantics -> distinct hash ---
def test_nonequivalent_graphs_hash_differ():
    g1 = make_graph({"x1": "linear", "x2": "relu", "x3": "linear"}, [("x1", "x2", 0), ("x2", "x3", 0)])
    g2 = make_graph({"x1": "linear", "x2": "gelu", "x3": "linear"}, [("x1", "x2", 0), ("x2", "x3", 0)])
    assert semantic_hash(g1, {"x3"}) != semantic_hash(g2, {"x3"})
    assert not graphs_equivalent(g1, {"x3"}, g2, {"x3"})

# --- Trap 1 regression: the bare call must never be trusted directly ---
def test_bare_isomorphism_trap_regression():
    g1 = make_graph({"x1": "linear", "x2": "relu", "x3": "linear"}, [("x1", "x2", 0), ("x2", "x3", 0)])
    g2 = make_graph({"x1": "conv", "x2": "sigmoid", "x3": "attention"}, [("x1", "x2", 0), ("x2", "x3", 0)])
    assert nx.is_isomorphic(g1, g2) is True          # the trap fires on the bare call
    assert graphs_equivalent(g1, {"x3"}, g2, {"x3"}) is False  # the labeled pipeline correctly rejects

# --- Direction 2, adversarial: a canonicaliser that unifies MORE must not
#     unify too much. Same op multiset, same degree sequence, NOT isomorphic. ---
def test_near_miss_pair_still_separates():
    a = make_graph({"in": "linear", "p1": "relu", "p2": "relu",
                    "q1": "sigmoid", "q2": "sigmoid", "out": "add"},
                   [("in", "p1", 0), ("in", "p2", 0), ("p1", "q1", 0),
                    ("p2", "q2", 0), ("q1", "out", 0), ("q2", "out", 0)])
    b = make_graph({"in": "linear", "p1": "relu", "p2": "relu",
                    "q1": "sigmoid", "q2": "sigmoid", "out": "add"},
                   [("in", "p1", 0), ("in", "p2", 0), ("p1", "q1", 0),
                    ("q1", "q2", 0), ("p2", "out", 0), ("q2", "out", 0)])
    assert not _labeled_isomorphic(a, b)             # genuinely different structures
    assert semantic_hash(a, {"out"}) != semantic_hash(b, {"out"})
    assert not graphs_equivalent(a, {"out"}, b, {"out"})

test_equivalent_graphs_hash_equal()
test_branch_and_merge_does_not_falsely_split()
test_deep_same_port_branches_do_not_falsely_split()
test_residual_passthrough_survives_canonicalisation()
test_nonequivalent_graphs_hash_differ()
test_bare_isomorphism_trap_regression()
test_near_miss_pair_still_separates()
print("equivalence + hash properties verified (both directions)")
```

## When the Hash and the Exact Check Disagree

`graphs_equivalent` can reach a state the happy path never exercises: hashes match, but the exact labeled-isomorphism check says the canonical forms are *not* isomorphic. Decide what this means before it happens, because it will be tempting to shrug at:

- It is **not** a tolerable inconsistency to log and move on from. Equal canonical bytes with non-isomorphic canonical graphs means either the serialization is lossy (two different graphs produce the same bytes — a serialization bug) or a genuine sha256 collision occurred (astronomically unlikely; assume the bug).
- Treat it as a **halting defect in the canonicaliser or serializer**, with the disagreeing pair preserved as a regression fixture. This state is one of the few places the pipeline can catch its own identity layer being wrong — wasting it on a warning log discards the highest-value bug report the system will ever generate.

The reverse disagreement — hashes differ but someone proves the graphs equivalent by hand — is the false-split case: same severity, same response, but *nothing will detect it automatically*. That is why the adversarial Direction-1 tests above (relabelings, port permutations, symmetric structures drawn from your own grammar) exist: they are the only detector.

Two things about that detector, learned the hard way by this pack's own reference implementation:

- **The fixture class has to be deeper than depth 1.** Port-swapped *single-node* branches — the v1 regression fixture — are passed by any canonicaliser with bidirectional refinement, so a suite built only from them issues a clean bill to a v2-class implementation that still false-splits. The discriminating fixture is repeated **multi-node** branches feeding the same port of a commutative merge, with the internal wiring permuted (`make_same_port_deep_branch_pair` above). Generate that pair at every branch depth your grammar can express, not just the depth someone happened to draw first.
- **Bidirectional refinement is necessary but not sufficient.** Reading the refinement loop and confirming it folds in both `predecessors()` and `successors()` establishes only that v1's bug is gone. The sufficient construction is orbit-aware individualization-refinement resolving the tie residue; a raw-ID tie-break after a correct bidirectional refinement is still a false-split generator. Audit the tie-break, not just the loop.

## Hash-Version Migration

When `HASH_VERSION` bumps, every persisted identity built on the old version becomes incomparable with new output. Two workable migration disciplines:

- **Recompute-on-read**: archive entries store the raw candidate alongside the hash; any entry read under a newer version gets rehashed and rewritten. Cheap, incremental, correct — but the archive is mixed-version until fully touched, so *every* cross-entry comparison must check version fields first.
- **Batch recompute**: a migration pass rehashes the whole archive under the new version before the new code serves traffic. Expensive up front, but comparisons never need version guards afterward.

Either is fine. What is not fine: comparing hashes without checking versions, or "grandfathering" old hashes because recomputing is inconvenient. An archive that mixes hash versions without discriminating is an archive whose dedup and diversity numbers mean nothing — and it got that way while every individual query still returned plausible-looking results.

Tie the bump to the change mechanically: a test that hashes a small fixture corpus and compares against stored golden values will fail the moment canonicalisation or serialization changes behavior — which is precisely the moment the version string must move. "We'll remember to bump it" is not a control; a failing golden-hash test is.

## Equivalence Classes and Provenance

An equivalence-class ID is the semantic hash, reused: every candidate that canonicalises to the same form shares a class ID, and the archive (see `lineage-mutation-and-recombination.md`) keys retrieval, deduplication, and diversity accounting on that class ID rather than on raw candidate identity. This means three distinct identities exist for every candidate — worth naming explicitly because code that conflates them produces confusing bugs:

1. **Raw identity** — the exact graph the generator emitted, before any canonicalisation.
2. **Canonical semantic identity** — the equivalence-class ID / semantic hash, stable across serialization and generator noise.
3. **Compiled/executable identity** (if a lowering stage exists downstream) — a specific device- and dtype-targeted implementation of the canonical semantics. Out of scope for this pack, but worth flagging: two different compiled artefacts may implement the same canonical semantic identity, and a compiled artefact must never be trusted to still implement that identity without independent conformance verification after compilation.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "`is_isomorphic()` is the standard networkx isomorphism check, it must be doing the right thing" | It checks topology only by default; operator labels require `node_match` explicitly |
| "We pinned the networkx version, so the hash function is stable" | Pinning helps, but the hash is still attribute-blind unless `node_attr` is passed — a version pin does not fix a missing argument |
| "Hash collision is astronomically unlikely, we don't need the isomorphism fallback" | Collision probability is about the hash function; it says nothing about bugs in canonicalisation, serialization, or the surrounding code — the fallback catches those too |
| "We tested that identical structures hash the same, that's the important direction" | Test it on *adversarial* identicals — relabelings, port permutations, symmetric branches — not just renamed chains; and the other direction is just as load-bearing |
| "Hashes disagree but the graphs look equivalent — probably fine, the hash is conservative" | A false split is invisible to every downstream check, forever; "conservative" here means "silently double-counting candidates" |
| "The canonicaliser change is small, old hashes are probably still valid" | "Probably still valid" across a canonicalisation change is exactly the silent invalidation the version field exists to prevent; bump it and migrate |
| "We'll bump the hash version when we remember to" | Golden-hash fixtures fail the build the moment behavior changes — that, not memory, is the control |

## Red Flags Checklist

- [ ] **`nx.is_isomorphic()` (or any topology-only isomorphism call) used anywhere near candidate equivalence** without `node_match`/`edge_match`
- [ ] **A library hash function used directly as the persisted identity**, with no owned serialization layer on top
- [ ] **No `hash_version` field**, or a version bump that isn't enforced by a golden-hash fixture test
- [ ] **Only one direction of the equivalence property tested** — or Direction 1 tested only on renamed chains, never on port permutations or symmetric structures
- [ ] **Direction-1 adversarial fixtures stop at depth-1 branches** — single-node parallel branches pass under a raw-ID tie-break; repeated multi-node branches on the same port are the pair that discriminates
- [ ] **The canonicaliser's tie-break was never audited**, only its refinement direction — bidirectional refinement is necessary, orbit-aware individualization is what makes it sufficient
- [ ] **Hash/exact-check disagreement handled as a log line** instead of a halting defect with the pair preserved
- [ ] **Archive mixes hash versions without version-checked comparisons**
- [ ] **Equivalence-class IDs computed from raw graphs**, not from the canonical form
- [ ] **Hash equality trusted as final** with no exact-isomorphism fallback anywhere in the codebase

## Diagnostic Questions

1. **What are the inputs to your hash, byte for byte?** If you cannot write down the exact serialization format, a library is deciding it for you, and the library's next version will decide differently.
2. **What is your hash version, and what test fails when someone forgets to bump it?** If the answers are "we don't have one" and "none," every future canonicaliser fix is a silent archive invalidation.
3. **Which adversarial pairs are in your Direction-1 test set, and how deep do they go?** Renamed chains catch nothing interesting; port permutations of single-node branches catch a v1-class bug and nothing more. Repeated multi-node branches on the same port of a commutative merge are the pair that separates a sufficient canonicaliser from a plausible one.
4. **What does your code do when the hash and the exact check disagree?** If nobody knows, the answer is "logs and continues," which discards the best bug report the identity layer will ever produce.
5. **Can you compare two archive entries from six months apart and know the comparison is valid?** If hash versions aren't stored and checked, you cannot.

## Cross-References

- **The canonical form this hash is built on, the bidirectional-refinement requirement, and the orbit-aware individualization that resolves the tie residue**: `canonicalisation-and-normal-forms.md`
- **Where this equivalence check plugs into the full legality gate.** Pipeline order across this pack is fixed: cheap legality checks (shape, type, acyclicity, contract arity) → canonicalisation → the full structural-verification gate, including reachability and dead-node detection, run on the canonical form. This hash identifies a candidate the gate has cleared, so it is computed once, on that canonical form: `structural-verification.md`
- **Why diversity must be measured in canonical/functional space using this hash**: `diversity-and-mode-collapse.md`
- **How the archive keys retrieval and dedup on equivalence-class ID**: `lineage-mutation-and-recombination.md`
- **The full anti-pattern catalogue**: `synthesis-anti-patterns.md`
