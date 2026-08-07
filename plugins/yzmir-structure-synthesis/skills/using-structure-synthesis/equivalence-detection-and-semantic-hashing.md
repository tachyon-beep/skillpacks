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

This pack's own reference implementation is a worked example of the consequence: its hash version is `structhash-v2` because `structhash-v1` was the predecessor-only design. The refinement change alters canonical bytes for branch-and-merge graphs, so every v1 hash is incomparable with every v2 hash — and the version field is what makes that incomparability *visible* instead of silent. A version bump on canonicaliser change is not bureaucracy; it is the only thing standing between "we fixed the canonicaliser" and "we corrupted the archive and called it a fix."

## The GREEN Fix: Own the Serialization, Version the Hash, Never Trust Equality Alone

Three disciplines, applied together:

1. **Canonicalise first** — the hash is computed over the *canonical form* (`canonicalisation-and-normal-forms.md`), never the raw graph. Two semantically identical raw graphs canonicalise to byte-identical serializations; the hash is deterministic from there.
2. **Pin your own serialization** — do not delegate byte layout to a library's internal iteration order or a general-purpose hash function's attribute handling. Serialize nodes and edges yourself, in a fixed, sorted order, with an explicit format.
3. **Version the hash** — a `hash_version` string is part of the hash input. When the canonicalisation algorithm, the serialization format, or a pinned dependency changes in a way that could change output, bump the version. Old archives keep their old hash and are recomputed under the new version rather than silently compared across an undeclared change.

An external graph hash function (Weisfeiler-Leman or otherwise) is legitimate as a **fast pre-filter** — cheap to compute, used to skip the expensive exact check for the overwhelming majority of non-matching pairs — but it is never the authority. Two candidates with equal fast-hash are still confirmed or refuted by the canonical-bytes hash (or, if you don't trust that either, by the exact labeled isomorphism check). Two candidates with unequal fast-hash are correctly assumed non-equivalent without further work, *provided* the fast hash's false-negative rate has actually been measured, not assumed to be zero.

## Executable Decision Procedure

The same `canonicalise` / `canonical_bytes` implementation from `canonicalisation-and-normal-forms.md` — bidirectional refinement, sha256 signature compression — extended with the versioned hash and an equivalence check that tests both directions:

```python
import hashlib
import networkx as nx

HASH_VERSION = "structhash-v2"  # v1 = predecessor-only refinement (Trap 3); bump on ANY
                                # canonicalisation or serialization change
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

def canonicalise(g, outputs):
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
                pruned.add_edge(pred, succ, in_port=pruned.edges[n, succ]["in_port"])
                pruned.remove_node(n)
                changed = True
    order = list(nx.topological_sort(pruned))
    signature = {n: _digest("op", pruned.nodes[n]["op"]) for n in order}
    for _ in range(len(order) + 1):
        new_sig = {}
        for n in order:
            incoming = sorted((signature[p], pruned.edges[p, n]["in_port"])
                              for p in pruned.predecessors(n))
            outgoing = sorted((signature[s], pruned.edges[n, s]["in_port"])
                              for s in pruned.successors(n))
            new_sig[n] = _digest(signature[n], incoming, outgoing)
        signature = new_sig
    ranked = sorted(order, key=lambda n: (signature[n], n))
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

# --- Direction 1, adversarial: port-asymmetric branches (Trap 3 regression) ---
def test_branch_and_merge_does_not_falsely_split():
    g1 = make_graph({"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
                    [("in", "a", 0), ("in", "b", 0), ("a", "out", 0), ("b", "out", 1)])
    g2 = make_graph({"in": "linear", "a": "relu", "b": "relu", "out": "residual_add"},
                    [("in", "a", 0), ("in", "b", 0), ("a", "out", 1), ("b", "out", 0)])
    assert semantic_hash(g1, {"out"}) == semantic_hash(g2, {"out"})
    assert graphs_equivalent(g1, {"out"}, g2, {"out"})

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

test_equivalent_graphs_hash_equal()
test_branch_and_merge_does_not_falsely_split()
test_nonequivalent_graphs_hash_differ()
test_bare_isomorphism_trap_regression()
print("equivalence + hash properties verified (both directions)")
```

## When the Hash and the Exact Check Disagree

`graphs_equivalent` can reach a state the happy path never exercises: hashes match, but the exact labeled-isomorphism check says the canonical forms are *not* isomorphic. Decide what this means before it happens, because it will be tempting to shrug at:

- It is **not** a tolerable inconsistency to log and move on from. Equal canonical bytes with non-isomorphic canonical graphs means either the serialization is lossy (two different graphs produce the same bytes — a serialization bug) or a genuine sha256 collision occurred (astronomically unlikely; assume the bug).
- Treat it as a **halting defect in the canonicaliser or serializer**, with the disagreeing pair preserved as a regression fixture. This state is one of the few places the pipeline can catch its own identity layer being wrong — wasting it on a warning log discards the highest-value bug report the system will ever generate.

The reverse disagreement — hashes differ but someone proves the graphs equivalent by hand — is the false-split case: same severity, same response, but *nothing will detect it automatically*. That is why the adversarial Direction-1 tests above (relabelings, port permutations, symmetric structures drawn from your own grammar) exist: they are the only detector.

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
- [ ] **Hash/exact-check disagreement handled as a log line** instead of a halting defect with the pair preserved
- [ ] **Archive mixes hash versions without version-checked comparisons**
- [ ] **Equivalence-class IDs computed from raw graphs**, not from the canonical form
- [ ] **Hash equality trusted as final** with no exact-isomorphism fallback anywhere in the codebase

## Diagnostic Questions

1. **What are the inputs to your hash, byte for byte?** If you cannot write down the exact serialization format, a library is deciding it for you, and the library's next version will decide differently.
2. **What is your hash version, and what test fails when someone forgets to bump it?** If the answers are "we don't have one" and "none," every future canonicaliser fix is a silent archive invalidation.
3. **Which adversarial pairs are in your Direction-1 test set?** Renamed chains catch nothing interesting; port permutations, symmetric branches, and interleaved emission orders catch real refinement bugs.
4. **What does your code do when the hash and the exact check disagree?** If nobody knows, the answer is "logs and continues," which discards the best bug report the identity layer will ever produce.
5. **Can you compare two archive entries from six months apart and know the comparison is valid?** If hash versions aren't stored and checked, you cannot.

## Cross-References

- **The canonical form this hash is built on, the bidirectional-refinement requirement, and the tie-break safety argument**: `canonicalisation-and-normal-forms.md`
- **Where this equivalence check plugs into the full legality gate**: `structural-verification.md`
- **Why diversity must be measured in canonical/functional space using this hash**: `diversity-and-mode-collapse.md`
- **How the archive keys retrieval and dedup on equivalence-class ID**: `lineage-mutation-and-recombination.md`
- **The full anti-pattern catalogue**: `synthesis-anti-patterns.md`
