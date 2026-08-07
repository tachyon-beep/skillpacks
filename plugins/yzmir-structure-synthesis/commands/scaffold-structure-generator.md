---
description: Scaffold a structure generator + verifier + canonicaliser skeleton with failing-first tests for round-trip, canonicalisation idempotence, and hash equivalence in both directions
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<project-name> [--representation=pointer-based|edge-list|latent-decoder]"
---

# Scaffold Structure Generator

Create a new structure-synthesis pipeline with the module separation and failing-first test discipline this domain requires: a generator that never filters its own output, a canonicaliser that never becomes a generator, a verifier that consumes no utility signal, and a hasher whose stability is tested in both directions before anything else is built on top of it.

## What Gets Created

```
project-name/
├── grammar.py              # Operator whitelist, shape rules, ceilings
├── generator.py            # Emits raw candidates -- no self-filtering
├── canonicalizer.py        # Semantics-preserving normal form only
├── hasher.py                # Versioned semantic hash over the canonical form
├── verifier.py              # Legality gate -- consumes NO utility/reward/provenance
├── archive.py                # Lineage storage, keyed on canonical identity, retains failures
├── tests/
│   ├── test_round_trip.py            # encode(decode(x)) == x, and vice versa
│   ├── test_canonicalisation.py      # idempotence + order-independence + false-split regression
│   ├── test_hash_equivalence.py      # both directions: equal-semantics/equal-hash,
│   │                                  # distinct-semantics/distinct-hash
│   ├── test_verifier_boundary.py     # verifier signature accepts no utility/reward input
│   └── test_generator_boundary.py    # generator never returns fewer candidates than requested
├── requirements.txt
└── README.md
```

This layout is non-negotiable in spirit — file names may change, but the categories may not be omitted. Each file represents a discipline this pack documents:

| File | Discipline |
|------|------------|
| `grammar.py` | `typed-graph-grammars.md` |
| `generator.py` | `generation-strategies.md`, `learning-objectives-for-generators.md` |
| `canonicalizer.py` | `canonicalisation-and-normal-forms.md` |
| `hasher.py` | `equivalence-detection-and-semantic-hashing.md` |
| `verifier.py` | `structural-verification.md` |
| `archive.py` | `lineage-mutation-and-recombination.md` |

## Key Principles

### 1. Tests Are Failing-First

Every test file starts as a **known-failing** stub against the scaffold's initial no-op implementations — the point is to write the test before the implementation exists, so the implementation is built to satisfy a test that was already articulated, not retrofitted to whatever the implementation happens to do.

```python
# tests/test_canonicalisation.py — starts failing, stays in the suite permanently
def test_idempotence():
    g = make_example_graph()
    outputs = {"out"}
    c1 = canonicalise(g, outputs)
    c2 = canonicalise(c1, outputs=infer_outputs(c1))
    assert canonical_bytes(c1) == canonical_bytes(c2), "canonicalisation is not idempotent"

def test_order_independence():
    g1, g2 = make_example_graph(), make_reordered_equivalent_graph()
    assert canonical_bytes(canonicalise(g1, {"out"})) == canonical_bytes(canonicalise(g2, {"out"}))

def test_no_false_split_on_symmetric_structures():
    # Port-asymmetric pair: identical branches feeding a merge, port assignment
    # swapped between the two copies. Isomorphic graphs MUST canonicalise
    # identically -- a split here is invisible to every downstream check.
    g1, g2 = make_port_swapped_symmetric_pair()
    assert canonical_bytes(canonicalise(g1, {"out"})) == canonical_bytes(canonicalise(g2, {"out"}))
```

Reference implementation for these functions, including why the refinement must fold in both in-edges and out-edges: `canonicalisation-and-normal-forms.md`'s executable decision procedure.

### 2. Hash Equivalence Tested in BOTH Directions

```python
# tests/test_hash_equivalence.py
def test_equal_semantics_equal_hash():
    g1, g2 = equivalent_but_differently_labeled_graphs()
    assert semantic_hash(g1, out1) == semantic_hash(g2, out2)

def test_distinct_semantics_distinct_hash():
    g1, g2 = structurally_similar_but_semantically_different_graphs()
    assert semantic_hash(g1, out1) != semantic_hash(g2, out2)

def test_bare_isomorphism_is_not_used_anywhere():
    # grep-based regression guard: nx.is_isomorphic without node_match must not
    # appear anywhere near candidate-equivalence code. Note: this is a plain
    # substring grep followed by a Python-side filter, not a single regex with
    # negative lookahead -- `(?!...)` is a PCRE feature that POSIX `grep -E`
    # does not support (it fails silently on most greps rather than erroring,
    # which would make this guard always report zero findings).
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "is_isomorphic(", "."],
        capture_output=True, text=True,
    )
    suspicious = [l for l in result.stdout.splitlines() if "node_match" not in l]
    assert not suspicious, f"bare isomorphism check found: {suspicious}"
```

A hash test suite that only checks one direction has only half-verified the hash. See `equivalence-detection-and-semantic-hashing.md` for why both directions fail differently and both matter.

### 3. Generator Boundary Is Structurally Enforced

```python
# tests/test_generator_boundary.py
def test_pool_size_matches_request_regardless_of_internal_scores():
    pool = generator.generate_pool(request, k=8)
    assert len(pool) == 8, "generator must never silently drop candidates by internal score"

def test_generator_signature_has_no_utility_filter_path():
    import inspect
    sig = inspect.signature(generator.generate_pool)
    assert "utility_threshold" not in sig.parameters
    assert "min_predicted_score" not in sig.parameters
```

Soft tests — they make the anti-pattern visible in CI rather than in the failure mode it causes. See `learning-objectives-for-generators.md`.

### 4. Verifier Boundary Is Structurally Enforced

```python
# tests/test_verifier_boundary.py
def test_verifier_signature_has_no_utility_or_provenance_input():
    import inspect
    sig = inspect.signature(verifier.verify)
    forbidden = {"reward", "predicted_utility", "generator_id", "source", "future_utility"}
    assert not (forbidden & set(sig.parameters)), \
        f"verifier consumes forbidden inputs: {forbidden & set(sig.parameters)}"
```

Static, structural check — same discipline as `yzmir-morphogenetic-rl`'s governor-invariant tests, applied to the generation/verification boundary this pack owns. See `structural-verification.md`.

### 5. Archive Retains Failures, Keyed on Canonical Identity

```python
# archive.py
@dataclass
class ArchiveEntry:
    canonical_hash: str        # from hasher.py — the retrieval key
    raw_candidate: RawGraph
    outcome: Literal["accepted", "rejected_structural", "rejected_downstream"]
    lineage: list[str]         # parent hashes, for audit -- never read by verifier/generator
```

No `winners_only` mode. See `lineage-mutation-and-recombination.md` on why survivorship bias in the archive corrupts future search.

## Post-Scaffold Checklist

Before generating a single real candidate:

1. **Run the round-trip test**: `pytest tests/test_round_trip.py -v`
2. **Run the canonicalisation tests**: `pytest tests/test_canonicalisation.py -v`
3. **Run the hash-equivalence tests, both directions**: `pytest tests/test_hash_equivalence.py -v`
4. **Run the boundary-enforcement tests**: `pytest tests/test_verifier_boundary.py tests/test_generator_boundary.py -v`
5. **Confirm the grammar's ceilings are all independently checked** — node, edge, parameter, memory — not just node count
6. **Confirm the archive schema has no `winners_only` code path**

If any of (1)–(6) fails, fix it before generating real candidates. A pipeline built on top of a broken canonicaliser or an unverified hash direction produces results that look fine and are quietly wrong.

## Anti-Patterns This Scaffold Prevents

| Anti-pattern | How the scaffold prevents it |
|---|---|
| Generator self-filters its pool | `test_generator_boundary.py` asserts returned pool size matches requested size |
| Verifier consumes reward/provenance | `test_verifier_boundary.py` structurally checks the function signature |
| Canonicaliser is non-idempotent or order-dependent | `test_canonicalisation.py` runs both properties as tests, not assumptions |
| Hash only tested in one direction | `test_hash_equivalence.py` requires both `test_equal_semantics_equal_hash` and `test_distinct_semantics_distinct_hash` |
| Bare unlabeled isomorphism used anywhere | `test_bare_isomorphism_is_not_used_anywhere` greps for it as a regression guard |
| Archive discards failed/rejected candidates | `ArchiveEntry.outcome` has no `winners_only` mode; rejected entries are first-class |

## Load Detailed Guidance

For the canonicaliser and hasher (the technical core, get these right first):
```
Load skill: yzmir-structure-synthesis:using-structure-synthesis
Then read: canonicalisation-and-normal-forms.md
Then read: equivalence-detection-and-semantic-hashing.md
```

For the verifier:
```
Then read: structural-verification.md
```

For the generator and its training objective:
```
Then read: generation-strategies.md
Then read: learning-objectives-for-generators.md
```

For the archive:
```
Then read: lineage-mutation-and-recombination.md
```
