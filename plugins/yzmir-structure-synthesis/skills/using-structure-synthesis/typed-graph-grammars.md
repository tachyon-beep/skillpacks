---
name: typed-graph-grammars
description: "Use when defining the operator whitelist, typing/shape rules, and node/edge/parameter/memory ceilings a generator may emit within - and when deciding whether to expand a grammar to a new expressiveness level, staged behind a reliability gate rather than opened all at once."
---

# Typed Graph Grammars

## When to Use

- Starting a structure-synthesis pipeline and defining what the generator is even allowed to produce
- Deciding whether to add a new operator, or a new topology pattern, to an existing grammar
- Debugging "the generator produced something that technically fits the whitelist but blew our compute budget"
- Reviewing a grammar-expansion proposal for whether it needs a staged reliability gate first

This sheet defines the space; `generation-strategies.md` covers how a generator samples within it; `structural-verification.md` covers how a candidate is checked against it after generation.

## Core Principle

**A grammar is a claim about what can be verified, not just what can be generated. Every operator you whitelist and every ceiling you set is a promise that `structural-verification.md` can actually check it, at a cost that scales with how the grammar grows — not a promise about what would be interesting to generate.**

The temptation runs the other way: grammars grow because "the generator could do more if we let it," and each individual addition looks small. What actually determines whether an addition is safe is not how expressive it is but how *checkable* it stays — see `search-space-evolution-and-explosion-control.md` for the growth discipline. This sheet is about designing the grammar correctly the first time: bounded, typed, and staged.

## What a Grammar Declares

| Element | Purpose |
|---|---|
| **Operator whitelist** | The only operations a generator may emit. Anything else is a forbidden operation (`structural-verification.md`) |
| **Typing and shape rules** | Per-operator input/output arity and shape-transformation rule, so shape inference is mechanical |
| **Node ceiling** | Maximum node count per candidate |
| **Edge ceiling** | Maximum edge count per candidate — independent of node ceiling, not implied by it |
| **Parameter ceiling** | Maximum learnable-parameter count, summed across the candidate |
| **Memory ceiling** | Maximum activation/working-memory footprint at declared batch size |
| **Interface contract** | Required input/output shape relationship (e.g., shape-preserving insertion) — see `conditioning-on-context-and-contracts.md` |

All five ceilings are independent constraints. A generator that respects one does not automatically respect the others.

## The RED Scenario: One Ceiling Checked, Four Assumed

A budget check that verifies node count and stops there:

```python
def check_budget_WRONG(g, max_nodes, max_params):
    return g.number_of_nodes() <= max_nodes  # edges and params are never checked
```

A candidate with 6 nodes, each fully connected to every other node, has only 6 nodes — comfortably under a node ceiling of 8 — but 30 directed edges. If each node is a `linear_16x16` layer, this candidate also carries far more parameters and far more compute than a 6-node chain would suggest. **Node count is a poor proxy for graph cost.** A verifier or grammar checker that only tracks node count will pass a dense, expensive graph that a linear-chain intuition never anticipated.

### The GREEN Fix

Check every declared ceiling independently:

```python
import networkx as nx

OP_PARAM_COUNT = {"linear_8x8": 72, "linear_16x16": 272, "relu": 0, "identity": 0}

def check_budget(g: nx.DiGraph, max_nodes: int, max_edges: int, max_params: int) -> bool:
    n_params = sum(OP_PARAM_COUNT.get(g.nodes[n]["op"], 0) for n in g.nodes)
    return (
        g.number_of_nodes() <= max_nodes
        and g.number_of_edges() <= max_edges
        and n_params <= max_params
    )

def dense_but_few_nodes_graph():
    g = nx.DiGraph()
    for i in range(6):
        g.add_node(i, op="linear_16x16")
    for i in range(6):
        for j in range(6):
            if i != j:
                g.add_edge(i, j)  # 30 edges from only 6 nodes
    return g

g = dense_but_few_nodes_graph()
MAX_NODES, MAX_EDGES, MAX_PARAMS = 8, 12, 2000

def check_budget_WRONG(g, max_nodes, max_params):
    return g.number_of_nodes() <= max_nodes

wrong = check_budget_WRONG(g, MAX_NODES, MAX_PARAMS)
correct = check_budget(g, MAX_NODES, MAX_EDGES, MAX_PARAMS)
assert wrong is True    # node-only check wrongly passes
assert correct is False # full check correctly rejects: 30 edges > 12
print(f"node-only check: {wrong} (wrong) | full ceiling check: {correct} (correct)")
```

Both node ceiling and edge ceiling need enforcing even in grammars where they seem correlated — a generator exploring the boundary of what it's allowed to do will find the combinations a designer didn't picture.

## Staged Expressiveness Levels

A grammar does not have to open its full expressiveness on day one, and in most cases should not. Three levels, each strictly more expressive than the last, form a natural staging:

1. **Universal envelope** — a single, shape-preserving parametric form (e.g., a low-rank update or a small gated residual block) with the whitelist reduced to whatever fills in that form's internals. Removes the need for a "blueprint" decision entirely; the generator only chooses parameters within a fixed structural template.
2. **Constrained genotype** — the generator chooses from a small enumerated set of structural knobs (width, stage count, activation family, gating pattern, normalization placement) inside a still-fixed overall topology. More expressive than level 1, still cheap to verify because the space of possible shapes is enumerable.
3. **Typed DAG over the whitelist** — the generator emits an arbitrary small graph over the full operator whitelist, subject to the ceilings and interface contract. Most expressive, most expensive to verify, and the level where `structural-verification.md`'s full checklist earns its cost.

**Expand from one level to the next only behind a reliability gate**: the current level's generator, verifier, and canonicaliser must be demonstrably reliable (low structural-rejection rate, low duplicate rate, no known canonicalisation-drift bugs) before the grammar grows to the next level. Expanding early doesn't make the generator smarter; it makes the search space bigger while the tooling that keeps it honest hasn't caught up. See `search-space-evolution-and-explosion-control.md` for the gate criteria in full.

## Worked Example: Sizing a Level-2 Constrained Genotype

A generator producing small feature-transform blocks, staged at level 2:

```
knobs = {
    "hidden_width":     {8, 16, 32},
    "stage_count":      {1, 2, 3},
    "activation":       {"relu", "gelu"},
    "normalization":    {"none", "pre_norm"},
}
```

This space has `3 × 3 × 2 × 2 = 36` distinct structural configurations — small enough to exhaustively enumerate for verification testing, large enough to give the generator real choice. Contrast with jumping straight to level 3 (arbitrary DAG over 6+ operators with up to 12 nodes): the reachable structural space is combinatorially larger, and the verifier's cost — cycle detection, shape inference, zero-influence proof, all run per-candidate — grows with it. Staging is what makes the level-2 space fully testable before the level-3 space is attempted at all.

## Red Flags Checklist

- [ ] **Only one ceiling enforced** (usually node count) while edges, parameters, or memory are assumed bounded by implication
- [ ] **No staged levels** — the grammar opened its full whitelist and topology freedom from day one
- [ ] **Grammar expanded without checking the previous level's rejection/duplicate rates**
- [ ] **Interface contract implicit** rather than a declared, checkable field
- [ ] **Operator whitelist and shape rules live in different places** (or only one of them exists), so a new operator can be added without also adding its shape rule

## Cross-References

- **How a generator actually samples within this grammar**: `generation-strategies.md`
- **The full legality gate this grammar's ceilings feed into**: `structural-verification.md`
- **Growing the grammar safely, and the reliability-gate criteria in full**: `search-space-evolution-and-explosion-control.md`
- **Encodings a generator can emit for this grammar**: `graph-representations-for-generation.md`
- **The interface contract referenced above**: `conditioning-on-context-and-contracts.md`
