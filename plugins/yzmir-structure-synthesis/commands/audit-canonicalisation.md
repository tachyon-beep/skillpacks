---
description: Adversarially review a canonicaliser/hasher for semantic drift, non-idempotence, hash instability, and equivalence false positives/negatives, with severity-rated findings
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "<path-to-canonicaliser-and-hasher-code>"
---

# Audit Canonicalisation

Adversarially review an existing canonicaliser and semantic-hash implementation against the properties `canonicalisation-and-normal-forms.md` and `equivalence-detection-and-semantic-hashing.md` require. This is the audit that catches the single most dangerous class of bug in a structure-synthesis pipeline: a canonicaliser or hasher that looks correct on the examples someone tried, and corrupts every downstream measurement silently the moment it meets an example nobody tried.

## Core Principle

**A canonicaliser and hasher are correct only if every rewrite they apply has a proof obligation, and both directions of the hash's equivalence claim are tested.** This audit does not trust a claim of correctness; it derives specific test cases designed to break each rule and runs them.

## Audit Process

### Phase 1: Locate the Canonicalisation Rules

Read every rewrite rule in the canonicaliser. For each one, ask: **what is the proof that this rewrite preserves semantics?**

- A rule pruning nodes by structural pattern (degree, fan-in/out) without checking operator identity → **immediate finding**, cite `canonicalisation-and-normal-forms.md`'s RED scenario, this is the exact bug shape.
- An identity/pass-through **splice** (`pred → n → succ` becomes `pred → succ`) that does not check whether `pred → succ` already exists → **immediate finding**. On a `networkx.DiGraph` the `add_edge` silently *overwrites* the existing edge instead of adding a parallel one, so `out = residual_add(a·port1, identity(a)·port0)` — which computes `2a` — canonicalises to something that computes `a`, and false-merges with a genuinely single-input `residual_add(a)`. Operator-identity proof satisfied, semantics changed anyway. Construct that three-node graph against the grammar's own identity-op set to confirm.
- A rule merging two nodes as "duplicates" — what proof establishes they compute the identical function of identical inputs, versus merely looking similar?
- A rule reordering or relabeling — confirm it's provably semantics-irrelevant (pure serialization, not a computation change).

For every rule that lacks a stated proof, construct a concrete counterexample graph where the rule's precondition holds but the rewrite changes the computed function. If you can construct one, that's a confirmed finding, not a suspicion.

### Phase 2: Test Idempotence and Order-Independence

Run, or if no test exists, construct and run:

```python
def audit_idempotence(canonicalise_fn, sample_graphs):
    failures = []
    for g, outputs in sample_graphs:
        c1 = canonicalise_fn(g, outputs)
        c2 = canonicalise_fn(c1, outputs=infer_outputs(c1))
        if canonical_bytes(c1) != canonical_bytes(c2):
            failures.append(g)
    return failures

def audit_order_independence(canonicalise_fn, sample_graph_pairs):
    """sample_graph_pairs: (g1, g2) where g2 is g1 with nodes/edges added in
    different order or relabeled -- same semantics, different raw form."""
    failures = []
    for g1, g2 in sample_graph_pairs:
        if canonical_bytes(canonicalise_fn(g1, out1)) != canonical_bytes(canonicalise_fn(g2, out2)):
            failures.append((g1, g2))
    return failures
```

Sample graphs should specifically include: the largest legal graph under the grammar's ceilings, graphs with declared identity ops present, graphs with disconnected/dangling nodes, graphs at every staged expressiveness level in use, and — critically — **two escalating classes of symmetric structure**, because the first one alone will hand a clean bill to a canonicaliser that still false-splits:

1. **Port-permuted single-node branches** — two identical one-op branches feeding *different* ports of a merge, with the assignment swapped between copies. Catches predecessor-only refinement.
2. **Relabeled/re-wired repeated MULTI-node branches feeding the SAME port of a commutative merge** — e.g. two `in → relu → sigmoid → merge` chains, both on port 0, with the mid-chain wiring swapped between copies (`make_same_port_deep_branch_pair` in `equivalence-detection-and-semantic-hashing.md`). Catches a *bidirectional* refinement whose leftover ties are broken by raw node ID: the relus are one tied orbit and the sigmoids another, and resolving the two independently picks a pairing that is not an automorphism. Generate this pair at **every branch depth and multiplicity the grammar can express**, not just depth 1.

Both classes are false-split detectors, and *nothing downstream will ever catch a split*, because the exact-isomorphism fallback only fires when hashes already agree. A canonicaliser tested only on chains, hand-picked examples, or class-1 pairs has not been tested on the cases most likely to reveal a bug.

While reading the refinement loop itself, check **two** things, and do not stop at the first:

- **Direction**: does each round fold in both `predecessors()` and `successors()` (with ports)? A one-directional fold is an immediate finding regardless of whether a failing pair has been constructed yet — see `canonicalisation-and-normal-forms.md` RED Scenario 2 for the mechanism.
- **Tie-break**: what resolves nodes that still share a signature at the refinement fixed point? **Bidirectional refinement is necessary but not sufficient.** If ties fall through to the raw node ID (`sorted(..., key=(signature, node_id))` or equivalent), that is an immediate finding of the same severity — construct the class-2 pair above against the grammar and it will split. The sufficient construction is orbit-aware individualization-refinement: individualize each member of the smallest tied cell, re-refine, recurse, take the lexicographic minimum, with the cell choice and the individualization both driven by *signature* rather than node ID. A budget guard on that search that silently falls back to a raw-ID tie-break instead of raising is the same finding wearing a disguise.

### Phase 3: Test Both Directions of Hash Equivalence

```python
def audit_hash_both_directions(semantic_hash_fn, equivalent_pairs, distinct_pairs):
    false_splits = [  # equal semantics, hash disagrees
        (g1, g2) for g1, out1, g2, out2 in equivalent_pairs
        if semantic_hash_fn(g1, out1) != semantic_hash_fn(g2, out2)
    ]
    false_merges = [  # distinct semantics, hash agrees
        (g1, g2) for g1, out1, g2, out2 in distinct_pairs
        if semantic_hash_fn(g1, out1) == semantic_hash_fn(g2, out2)
    ]
    return false_splits, false_merges
```

Both directions are Critical, for different reasons. A `false_merge` does more damage per instance — two different candidates silently treated as one, with one of them effectively discarded from the archive or pool. A `false_split` is *less* damaging per instance but **structurally undetectable downstream**: the exact-isomorphism fallback only runs when hashes agree, so nothing in the pipeline will ever surface a split on its own — this audit's adversarial pairs are the only detector it will ever meet.

### Phase 4: Check Hash Versioning and Serialization Ownership

- Is there a `hash_version` (or equivalent) field baked into the hash input? If not: **finding** — any future canonicalisation or serialization change silently invalidates every archived hash with no error.
- Is the byte serialization pinned and self-owned (explicit sorted node/edge iteration, explicit field format), or delegated to a library's default iteration order or hash function? If delegated: **finding**, cite `equivalence-detection-and-semantic-hashing.md`'s library-hash trap, and check whether the specific library/version in use has a documented behavior change (check the library's changelog/release notes for the versions spanning the codebase's dependency lock history).

### Phase 5: Grep for the Bare-Isomorphism Trap

```bash
grep -rn "is_isomorphic(" . | grep -v "node_match"
```

Any hit is a candidate finding — confirm by reading the call site: is it comparing candidate graphs (finding) or something unrelated (not a finding, e.g., comparing test fixtures with identical labels by construction)?

### Phase 6: Severity Rating

| Finding type | Severity | Why |
|---|---|---|
| Canonicaliser rewrite with no proof obligation, counterexample constructed | Critical | Silently corrupts every downstream measurement for affected candidates |
| Hash false merge (distinct semantics, same hash) | Critical | Silently discards or conflates distinct candidates |
| Non-idempotent canonicalisation | High | Candidate identity depends on how many times canonicalisation happens to run |
| Hash false split (equal semantics, different hash) | Critical | Inflates diversity and corrupts dedup — and no downstream check can ever detect it |
| Identity splice with no `has_edge(pred, succ)` guard on a `DiGraph` | Critical | Overwrites a real edge: semantics-changing rewrite *and* false merge, both silent |
| Refinement is bidirectional but ties fall through to raw node IDs | Critical | False-splits repeated multi-node branches on a commutative merge; passes every depth-1 fixture |
| No `hash_version` field | Medium | Not corrupting today, but a dependency/library upgrade will silently invalidate history with no warning |
| Bare `is_isomorphic()` found, confirmed used on candidates | Medium–Critical depending on where it's used | Topology-only match on typed structures is close to always wrong for this domain |
| Order-independence not tested (but not proven to fail) | Low–Medium | Untested, not necessarily broken — recommend adding the test |

## Output Format

```markdown
## Canonicalisation & Hashing Audit

### Scope
[Files/functions reviewed]

### Findings
For each finding:
- **Severity**: Critical / High / Medium / Low
- **Location**: file:line
- **Rule/property violated**: [which of the proof obligations or hash-direction tests]
- **Evidence**: [the counterexample graph, or the failing test output]
- **Fixing sheet**: `canonicalisation-and-normal-forms.md` or `equivalence-detection-and-semantic-hashing.md`

### Properties Verified Clean
[List what was checked and found correct — idempotence held on N sample graphs, both hash directions held on M pairs, etc. Specific numbers, not "looks fine"]

### Untested Properties
[What could not be verified — no test harness for order-independence, no access to the largest-legal-graph case, etc.]
```

A zero-findings audit is not automatically a clean result — if Phase 1–5 weren't all actually exercised (e.g., no sample graphs were available to test idempotence against), say so explicitly in "Untested Properties" rather than reporting a clean bill of health for a check that didn't run.

## Reference

For the properties this audit checks in full:
```
Load skill: yzmir-structure-synthesis:using-structure-synthesis
Then read: canonicalisation-and-normal-forms.md
Then read: equivalence-detection-and-semantic-hashing.md
```

For the full anti-pattern catalogue if the audit surfaces something broader than canonicalisation/hashing:
```
Then read: synthesis-anti-patterns.md
```

For a full-pipeline audit beyond just canonicalisation/hashing:
```
agent: synthesis-integrity-reviewer
```
