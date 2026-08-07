---
name: ir-contracts-and-semantic-identity
description: Use when defining what a tensor compiler may and may not change about an incoming graph IR — topology and dataflow semantics versus schedule, layout, fusion and kernels — and how to carry a semantic hash through compilation so the artifact's claim stays falsifiable.
---

# IR Contracts and Semantic Identity

## When to Use

- You are defining the interface between whatever produces your graph IR and whatever compiles it.
- A pass wants to insert, remove, or reorder a node and you need a principled answer to "is that allowed?"
- You need to detect, mechanically, that a pass pipeline changed meaning.
- Someone proposes that the compiler "canonicalise" or "clean up" the incoming IR.

---

## Core Principle

**The compiler is handed a program and a name for that program. It may change how the program runs. It may not change what the name refers to.**

The name is the semantic hash. It arrives with the IR, it is not computed by the compiler, and it is copied onto the artifact untouched. That copy is a *claim*: "this artifact implements the program identified by this hash." A claim you can check is worth having. A hash the compiler computed from its own output is a tautology.

### The failure this prevents

Without an explicit contract, "improvements" accumulate that individually seem harmless and collectively change the program:

- A lowering pass replaces `x.sigmoid()` with `1/(1+exp(-x))` — mathematically identical, numerically different at the tails, and the backward has different overflow behaviour.
- A cleanup pass removes a node whose output "isn't used" — except it was a scaled residual whose contribution was zero at initialisation and nonzero after training.
- A fusion pass reassociates `(a+b)+c` to `a+(b+c)` — legal in ℝ, not legal in float32, and the reference run now disagrees with the compiled run by more than the declared tolerance.

Each of those is defensible in isolation. Together they mean the deployed program is not the approved program, and — this is the part that hurts — **there is no record of the divergence**, because nobody wrote down what was off-limits.

---

## The May-Change / Must-Not-Change List

Write this table for your IR. This one is a good default.

| Category | Compiler MAY change | Compiler MUST NOT change |
|----------|---------------------|--------------------------|
| **Topology** | — | Add, remove, duplicate, or re-parent a semantic node |
| **Dataflow** | — | Which value feeds which consumer |
| **Op identity** | Lower one IR op to several target ops (a *declared* decomposition) | Substitute a mathematically-similar op not in the decomposition table |
| **Schedule** | Execution order, subject to dependencies | Order where it is observable (RNG consumption, in-place aliasing) |
| **Layout** | Strides, memory format, padding, tiling | Logical shape or the input/output contract |
| **Fusion** | Merge adjacent ops into one kernel | Fuse across a node whose intermediate the contract exposes |
| **Kernels** | Any kernel meeting the declared numerical contract | A kernel outside the contract (lower precision, nondeterministic when determinism is declared) |
| **Constants** | Fold constant subgraphs at compile time | Fold anything reachable from a parameter that may still be trained |
| **Memory** | Buffer reuse, in-place where provably safe, checkpointing | Reuse a buffer still live under any legal execution order |
| **Precision** | Accumulate wider than declared | Compute narrower than declared |

The asymmetry in the last row is the useful shape: **the compiler may always be more careful than the contract, never less.** Accumulating a float32 reduction in float64 is fine. Doing it in bfloat16 is a contract violation even if this particular test passes.

### Decompositions are a declared list, not a judgement call

"Lower one op to several" is legal *only* against a table that both sides can see:

```python
DECOMPOSITIONS = {
    # ir_op: (target_ops, backward_equivalent, justification)
    "gelu_tanh": (["tanh", "mul", "add", "pow"], True,
                  "exact tanh-approximation form; backward derived by autograd"),
    "layer_norm": (["mean", "var", "sub", "div", "mul", "add"], True,
                   "matches ATen decomposition; eps applied inside sqrt"),
}
```

`backward_equivalent` is the field everyone forgets. A decomposition whose forward matches and whose backward differs is the most common semantic drift in this domain, because forward-only conformance never sees it. See `conformance-testing.md` and `operator-lowering-and-kernel-selection.md`.

---

## Computing a Semantic Hash

A semantic hash must be invariant to things that are not semantics (node names, whitespace) and sensitive to everything that is (op, target, dataflow edges, literal arguments).

Emission order of *independent* nodes is the case to be honest about. The implementation below identifies nodes positionally, which buys rename-invariance and costs order-invariance: two graphs that compute the same thing but emit independent nodes in different orders hash differently (verified on torch 2.9.1). **Order-invariance is bought upstream, not here** — canonicalise emission order in the producer (a deterministic topological sort with a stable tiebreak) before the hash runs. If you instead canonicalise inside the hash, every stored hash you already have is invalidated, so make that choice once, at the start.

This is verified against `torch.fx`, and generalises to any graph IR with the same three concepts (op kind, target, positional/keyword arguments):

```python
import hashlib, json
from collections import Counter
import torch, torch.fx as fx

SEMANTIC_OPS = {"placeholder", "call_function", "call_method", "call_module", "output"}

def semantic_skeleton(gm: fx.GraphModule) -> list[dict]:
    """What the graph COMPUTES, stripped of what it is CALLED.

    Node identity is positional (index in topological order), so renaming a node
    cannot change the hash. `get_attr` is excluded: parameter *values* are not
    semantics, they are state — hash them separately if your contract needs it.

    Two limits of positional identity, both measured on torch 2.9.1 — know them
    before you rely on this hash as a cache key:

    1. Emission order IS significant. Independent nodes emitted in a different
       order produce a different hash. Canonicalise order in the producer.
    2. Indices are assigned over ALL nodes, including the excluded `get_attr`s,
       so moving a state node shifts every semantic index after it. Enumerate
       only over SEMANTIC_OPS if you want indices insensitive to state
       placement — and re-baseline every stored hash if you make that change.
    """
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    skeleton = []
    for n in gm.graph.nodes:
        if n.op not in SEMANTIC_OPS:
            continue
        target = n.target if isinstance(n.target, str) else getattr(
            n.target, "__name__", str(n.target))
        skeleton.append({
            "i": order[n],
            "op": n.op,
            "target": target,
            "args": [order[a] if isinstance(a, fx.Node) else repr(a) for a in n.args],
            "kwargs": {k: (order[v] if isinstance(v, fx.Node) else repr(v))
                       for k, v in sorted(n.kwargs.items())},
        })
    return skeleton

def semantic_hash(gm: fx.GraphModule) -> str:
    payload = json.dumps(semantic_skeleton(gm), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
```

Two design decisions worth stating, because both are commonly got wrong:

- **Node names are excluded.** Positional indices mean a pass that renames `relu` to `relu_1` does not change the hash. If names were included, every graph round-trip would look like a semantic change and the hash would be useless — people would stop checking it.
- **Parameter values are excluded.** A trained and an untrained copy of the same architecture have the same *semantics* and different *state*. Merging them means every optimiser step invalidates every artifact in your cache. If your contract genuinely needs weight identity, hash it as a separate field.

---

## Detecting a Contract Violation Mechanically

The hash tells you *that* something changed. This tells you *what*:

```python
def contract_violations(before: fx.GraphModule, after: fx.GraphModule) -> list[str]:
    """Report semantic nodes invented or removed by a pass pipeline.

    Legal optimisations (fusion into a call_module, layout changes, scheduling)
    are visible here too, which is the point: they must be DECLARED in the
    manifest, not silently absorbed. See compilation-manifests-and-reproducibility.md
    """
    kind = lambda gm: Counter((e["op"], e["target"]) for e in semantic_skeleton(gm))
    b, a = kind(before), kind(after)
    findings = []
    for k, n in (a - b).items():
        findings.append(f"INVENTED semantic node: {k[0]}:{k[1]} (+{n})")
    for k, n in (b - a).items():
        findings.append(f"REMOVED semantic node: {k[0]}:{k[1]} (-{n})")
    return findings
```

Run it after every pass in development, and in CI over the whole pipeline. Its output is not automatically a failure — a fusion pass legitimately removes two nodes and adds one — but every line it prints must correspond to a **declared** manifest entry. Anything it prints that the manifest does not explain is a bug.

That is the operational form of "every optimisation is traceable": the diff and the manifest must reconcile, line for line.

---

## Worked Example (verified against torch 2.9)

```python
import torch, torch.nn as nn, torch.fx as fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
    def forward(self, x):
        h = self.lin(x)
        return torch.relu(h) + h          # residual: h is used TWICE

gm = fx.symbolic_trace(M())
print(semantic_hash(gm))                   # e.g. 670e52cd31a40d89

# A pass that inserts an extra relu — "harmless, relu is idempotent"
gm_bad = fx.symbolic_trace(M())
relu_node = next(n for n in gm_bad.graph.nodes
                 if n.op == "call_function" and getattr(n.target, "__name__", "") == "relu")
with gm_bad.graph.inserting_after(relu_node):
    extra = gm_bad.graph.call_function(torch.relu, (relu_node,))
relu_node.replace_all_uses_with(extra)
extra.args = (relu_node,)
gm_bad.graph.lint(); gm_bad.recompile()

print(contract_violations(gm, gm_bad))
# ['INVENTED semantic node: call_function:relu (+1)']
print(semantic_hash(gm_bad) != semantic_hash(gm))   # True
```

`relu(relu(x)) == relu(x)`, so the *outputs* are bit-identical and every forward conformance test passes. The violation is still real: the artifact no longer corresponds to the approved graph, the extra node costs a kernel launch, and — crucially — the next engineer who reads the artifact's graph sees a program nobody approved. Idempotence made the symptom invisible; the contract check found it anyway.

This is why the contract check is separate from the numerical check. **Numerical agreement is necessary and not sufficient.**

---

## Ingest Must Not Repair

A tempting stage-zero convenience: normalise the incoming IR — dedupe constants, canonicalise argument order, drop no-ops — before compiling.

Do not do it inside the compiler. If ingest rewrites the graph, then `spec.semantic_hash` describes the pre-repair graph while the compiler consumed the post-repair graph, and the artifact's claim is about a program that was never compiled. When the gate later disagrees, the diff points at the repair, and the repair is invisible because it happened before the manifest started.

If normalisation is genuinely needed, it belongs **upstream, in the producer, before the hash is taken.** The rule is mechanical: everything that changes the graph must happen either before hashing or be recorded in the manifest. There is no third place.

---

## Executable Decision Procedure

When a pass wants to change the graph, answer in order. The first "no" stops you.

```python
def may_i_change_this(change: dict) -> tuple[bool, str]:
    """change = {kind, node_op, declared_in_manifest, in_decomp_table,
                 backward_equivalent, exposed_intermediate, narrows_precision}"""
    if change["kind"] in ("insert_semantic_node", "remove_semantic_node"):
        if not change.get("in_decomp_table"):
            return False, ("Topology change outside the decomposition table. "
                           "Deciding what the program IS belongs to the graph "
                           "producer, not to the compiler.")
        if not change.get("backward_equivalent"):
            return False, ("Decomposition with a non-equivalent backward. Declare it "
                           "in the numerical contract or reject. See conformance-testing.md")
    if change["kind"] == "fuse" and change.get("exposed_intermediate"):
        return False, ("Fusing across an intermediate the IO contract exposes. "
                       "The contract promised that value is observable.")
    if change.get("narrows_precision"):
        return False, ("Compiler may compute WIDER than the contract, never narrower. "
                       "See numerical-contracts-and-tolerances.md")
    if not change.get("declared_in_manifest"):
        return False, ("Legal change, but unrecorded. Add the manifest entry. "
                       "See compilation-manifests-and-reproducibility.md")
    return True, "Permitted and traceable."
```

Note that the last check fails an otherwise-legal change. That is deliberate: a legal optimisation nobody recorded is indistinguishable, six months later, from an illegal one.

The error text on the first branch names the real issue — deciding *what the program is* belongs to the component that generates graphs, not the one that compiles them. A compiler that starts inventing nodes to improve predicted performance has quietly promoted itself to architecture search.

---

## RED → GREEN Scenario

**RED.** A pipeline compiling generated architectures adds a "dead code elimination" pass. It removes nodes with no downstream consumers. One generated candidate contains a branch scaled by a coefficient that is zero at initialisation and becomes nonzero during training. DCE sees a subgraph multiplied by zero, folds it away, and the artifact is a strictly smaller network.

Every conformance test passes — at initialisation, the outputs *are* identical. The candidate is approved on the strength of a trial in which the branch could never contribute. It is deployed, trains, and the branch that was supposed to grow does not exist. The measured result is attributed to the architecture rather than to the compiler having deleted part of it.

The tell was available on day one and nobody looked: `contract_violations()` would have printed `REMOVED semantic node` for every folded op, and no manifest entry explained them.

**GREEN.**

1. The IR contract adds `trainability_mask` to `CanonicalSpec.input_output_contract` — which constants are frozen and which are parameters that may change.
2. Constant folding is restricted to values reachable only from frozen constants (`fusion-and-memory-planning.md`).
3. CI runs `contract_violations()` over the full pipeline and requires every line to reconcile against a manifest entry.
4. Conformance evaluates at the declared inputs **and** a perturbed-parameter point, so a zero-at-init branch is exercised.

Step 4 generalises: **conformance inputs must include a state where every part of the program is live.** Testing only at initialisation tests only the subgraph that happens to be active there.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Compiler computes the semantic hash | Tautology; claim is unfalsifiable | Carry it from the spec |
| Hash includes node names | Every round-trip looks like drift; people stop checking | Positional indices |
| Hash includes parameter values | Every optimiser step invalidates the cache | Hash state separately |
| Ingest "repairs" the IR | The hash describes a graph you did not compile | Normalise upstream, before hashing |
| Undeclared op substitution ("same maths") | Different numerics, different backward | Decomposition table with `backward_equivalent` |
| DCE without a trainability mask | Deletes branches that are zero *now* | Fold only from frozen constants |
| Contract violations checked but never reconciled with the manifest | Legal and illegal changes look identical | Require line-for-line reconciliation |

---

## Checklist

- [ ] A written may-change / must-not-change table exists for your IR
- [ ] `semantic_hash` is produced upstream and copied, never recomputed by the compiler
- [ ] The hash is invariant to node naming and to parameter values
- [ ] A decomposition table exists, with `backward_equivalent` recorded per entry
- [ ] `contract_violations()` (or equivalent) runs in CI over the whole pipeline
- [ ] Every violation line reconciles against a manifest entry
- [ ] Ingest does not rewrite the graph
- [ ] Constant folding respects a trainability mask
- [ ] Conformance inputs exercise parts of the graph that are inactive at initialisation

---

## Related Sheets

- [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — where identity sits in the pipeline
- [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — the precision half of the contract
- [conformance-testing.md](conformance-testing.md) — proving the artifact honours the contract
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — fusion legality in detail
- [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — the hash as a cache key
