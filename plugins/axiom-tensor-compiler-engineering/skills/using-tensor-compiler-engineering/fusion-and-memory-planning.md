---
name: fusion-and-memory-planning
description: "Use when deciding whether a fusion is legal, optimising layout, constant folding, or planning memory and buffer reuse in a tensor compiler \u2014 and when establishing how each optimisation earns its entry in the compilation manifest."
---

# Fusion and Memory Planning

## When to Use

- A fusion pass is about to merge nodes and you need a legality rule, not an intuition.
- Layout/memory-format decisions are changing results.
- Constant folding is removing more than you expected.
- Buffer reuse produces intermittent wrong answers.

---

## Core Principle

**Every optimisation in this sheet trades a guarantee for speed. The trade is only sound if the guarantee was one the contract said you could spend — and only auditable if the trade is written down.**

Fusion spends materialised intermediates. Layout optimisation spends stride assumptions. Constant folding spends the ability of a value to change later. Memory planning spends buffer independence. Each is fine when the contract permits it, and each is a miscompile when it does not.

### The failure this prevents

An optimisation that fires but is not recorded makes every future behavioural difference un-diagnosable. Two artifacts, same semantic hash, different numbers. If the manifest says "fused 14 ops" for one and nothing for the other, the investigation takes ten minutes. If neither manifest mentions fusion, the investigation is a bisection of the entire compiler across two toolchains — days, and often abandoned with "probably nondeterminism".

This is why the manifest requirement is in this sheet rather than only in the manifest sheet. **The moment to record an optimisation is the moment it fires**, and the optimisation passes are the ones with the strongest incentive to skip it.

---

## Fusion Legality

Five conditions. A fusion is legal only if all five hold.

```python
def fusion_legal(producer, consumer, contract, exposed_values, gm) -> tuple[bool, str]:
    # 1. Single consumer — otherwise the intermediate must still be materialised
    if len(producer.users) != 1:
        return False, (f"{producer.name} has {len(producer.users)} users; fusing would "
                       "either duplicate the computation or drop a value another "
                       "consumer needs")

    # 2. Not part of the declared output contract
    if producer.name in exposed_values:
        return False, (f"{producer.name} is exposed by the IO contract — the contract "
                       "promised this value is observable")

    # 3. No aliasing / in-place hazard
    if is_inplace(producer) or is_inplace(consumer):
        return False, ("in-place op in the fusion region: memory dependencies are not "
                       "graph edges. Functionalise first "
                       "(torch-fx-capture-and-transformation.md)")

    # 4. Reassociation permission, if the fusion reorders a reduction
    if reorders_reduction(producer, consumer) and not contract.reassociation_allowed:
        return False, ("fusion reorders a reduction and the numerical contract "
                       "forbids reassociation")

    # 5. Combined numerics stay inside the op's budget
    if fused_error_estimate(producer, consumer, contract) > contract.budget_for(consumer):
        return False, ("fused accumulation exceeds the declared budget — usually the "
                       "fused kernel accumulating in compute_dtype rather than "
                       "accumulate_dtype")
    return True, "legal"
```

Condition 1 is the one that bites, because multi-use intermediates are common and invisible in source. Verified on torch 2.9:

```python
class M(nn.Module):
    def forward(self, x):
        h = self.l(x)
        return torch.relu(h) + h        # h used TWICE — a residual

gm = fx.symbolic_trace(M())
{n.name: len(n.users) for n in gm.graph.nodes}
# {'x': 1, 'l': 2, 'relu': 1, 'add': 1, 'output': 0}
```

`l` has two users. Fusing `l → relu` into one kernel without materialising `h` breaks the residual: the `add` has nothing to add. `len(node.users)` is a two-character check that prevents an entire class of miscompile, and it is skipped constantly because in source the residual reads as one expression.

Condition 5 is the one that produces *plausible* wrong answers rather than crashes. A fused reduction that accumulates in the compute dtype instead of the accumulate dtype gives results that look approximately right — see the RED scenario in `numerical-contracts-and-tolerances.md`.

---

## Layout Optimisation

Layout — strides, memory format, padding, tiling — is representation, not semantics. Changing it is always legal *in principle* and frequently wrong *in practice*, because passes downstream assume contiguity.

Rules that keep it safe:

- **Layout changes never alter logical shape.** If a layout pass changes `shape`, it is not a layout pass.
- **Every layout decision is recorded** with the op it applies to. A CPU artifact choosing contiguous and a CUDA artifact choosing channels-last is expected; the manifest is how anyone knows.
- **Layout agreement is exact exactly when the kernels are the same.** Strided/non-contiguous views of an unchanged memory format run the same kernels and must agree bit-identically — measured `0.000e+00` on torch 2.9.1 across CPU and CUDA (`conformance-testing.md`). A memory-format change (`channels_last`) legally re-selects kernels, and different algorithms reassociate: measured up to `9.5e-07` on CPU and `4.1e-05` on CUDA with default TF32 for the same conv stack. So the rule is two-sided: a nonzero same-kernel difference is a bug with no noise floor, and a format-change difference must stay inside the reassociation budget *and* reconcile to a recorded kernel choice in the manifest.
- **Test the non-contiguous case.** A stride bug is invisible on `torch.randn` outputs, which are always contiguous. Sliced, transposed, and expanded tensors are where it appears.

---

## Constant Folding

Folding evaluates a subgraph at compile time and replaces it with its value. The legality question is entirely: **can this value change later?**

```python
def foldable(node, trainability_mask, gm) -> tuple[bool, str]:
    if node.op == "placeholder":
        return False, "graph input"
    if node.op == "get_attr":
        if trainability_mask.get(node.target, True):
            return False, (f"{node.target} is trainable — folding it bakes in the "
                           "CURRENT value and silently freezes the parameter")
        return True, "frozen constant"
    if node.op in ("call_function", "call_method", "call_module"):
        if is_nondeterministic(node) or has_side_effects(node):
            return False, "nondeterministic or side-effecting"
        return all(foldable(a, trainability_mask, gm)[0]
                   for a in node.args if isinstance(a, fx.Node)), "all inputs foldable"
    return False, "unknown node kind — default deny"
```

The `trainability_mask` check is what stops the RED scenario in `ir-contracts-and-semantic-identity.md`, where a branch scaled by a coefficient that is zero *at initialisation* gets folded away and the network deploys without it.

Two rules follow, and both are counterintuitive enough to state explicitly:

- **A value being zero right now is not evidence it is constant.** Folding must consult the trainability mask, never the current tensor contents.
- **Default deny.** An unrecognised node kind is not foldable. A folding pass that defaults to "probably fine" removes things nobody chose to remove.

---

## Memory Planning

Buffer reuse assigns two tensors the same storage when their lifetimes do not overlap. It is the highest-leverage memory optimisation and the easiest to get catastrophically wrong, because the failure is intermittent.

```python
def compute_lifetimes(gm) -> dict:
    """last_use must be over ALL legal execution orders, not the current one."""
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    return {n: (order[n], max((order[u] for u in n.users), default=order[n]))
            for n in gm.graph.nodes}

def reuse_legal(a, b, lifetimes, aliases, gm) -> tuple[bool, str]:
    (a_def, a_last), (b_def, b_last) = lifetimes[a], lifetimes[b]
    if not (a_last < b_def or b_last < a_def):
        return False, "lifetimes overlap"
    if aliases.get(a) or aliases.get(b):
        return False, ("tensor participates in a view/alias relationship; its true "
                       "lifetime extends to the last use of every alias")
    if is_graph_output(a, gm) or is_graph_output(b, gm):
        return False, "graph output — lifetime extends past the graph"
    if needs_saving_for_backward(a) or needs_saving_for_backward(b):
        return False, ("saved for backward: forward-only lifetime analysis is wrong "
                       "for any tensor the backward consumes")
    return True, "disjoint lifetimes, no aliases"
```

The last two conditions are where real bugs live.

**Views and aliases.** `y = x.view(...)` does not copy. If lifetime analysis tracks `x` and not `y`, `x`'s buffer is reused while `y` still points into it. The symptom is a wrong answer that depends on allocation order — intermittent, unreproducible, and typically blamed on hardware.

**Saved-for-backward tensors.** A forward-only lifetime analysis says an activation dies at its last forward use. It does not: the backward reads it, possibly thousands of operations later. Plan lifetimes over the **joint forward+backward graph** (`torch-compile-and-aotautograd.md`), not the forward alone. This is the memory-planning bug that only appears in training, which means it survives every inference test.

---

## Every Optimisation Earns a Manifest Entry

Uniform shape, appended at the moment the optimisation fires:

```python
def record(manifest, *, kind, nodes_before, nodes_after, legality_basis,
           numerical_impact, condition):
    manifest.append({
        "kind": kind,                       # "fusion" | "layout" | "const_fold" | "buffer_reuse"
        "nodes_before": nodes_before,       # names — makes the topology diff reconcilable
        "nodes_after": nodes_after,
        "legality_basis": legality_basis,   # which condition permitted it
        "numerical_impact": numerical_impact,  # "none" | "reassociation" | "widened_accum"
        "condition": condition,             # what made it fire: shape, device, dtype
    })
```

`condition` is what makes the manifest answer the hard question. When artifact A fuses and artifact B does not, `condition: "device=cuda, shape=(2,3,16,16)"` explains it in one line. Without it, the difference is a mystery with two artifacts and no lead.

`legality_basis` is the audit trail for correctness rather than performance: it lets a reviewer check that the fusion which turned out to be wrong claimed a condition that did not actually hold.

And the reconciliation rule, restated from `ir-contracts-and-semantic-identity.md`: **the topology diff between input IR and final graph must be explainable, line for line, by manifest entries.** Anything unexplained is a bug — that is the entire point of recording `nodes_before` / `nodes_after`.

---

## Executable Decision Procedure

```python
def optimisation_review(opt, contract, gm, exposed, trainability) -> tuple[str, str]:
    if opt.kind == "fusion":
        ok, why = fusion_legal(opt.producer, opt.consumer, contract, exposed, gm)
        if not ok:
            return "REJECT", why
    elif opt.kind == "const_fold":
        ok, why = foldable(opt.node, trainability, gm)
        if not ok:
            return "REJECT", why
    elif opt.kind == "buffer_reuse":
        ok, why = reuse_legal(opt.a, opt.b, opt.lifetimes, opt.aliases, gm)
        if not ok:
            return "REJECT", why
    elif opt.kind == "layout":
        if opt.changes_logical_shape:
            return "REJECT", "layout pass changed logical shape — that is not layout"

    if not opt.recorded_in_manifest:
        return "REJECT", ("legal but unrecorded. An unrecorded optimisation makes every "
                          "future behavioural difference un-diagnosable.")
    if opt.numerical_impact != "none" and not contract.reassociation_allowed:
        return "REJECT", "numerical impact without contract permission"
    return "ACCEPT", f"legal via {opt.legality_basis}, recorded"
```

A legal optimisation is rejected for being unrecorded. That is deliberate and it is the sheet's central discipline: in six months, an unrecorded legal optimisation and an unrecorded illegal one are the same object.

---

## RED → GREEN Scenario

**RED.** A memory planner reduces peak memory 30% by reusing buffers whose lifetimes do not overlap in the forward graph. Inference conformance passes on every input. Training is enabled and loss becomes erratic — not `nan`, just worse, intermittently, more often at larger batch sizes.

The investigation blames the learning rate, then the data pipeline, then a suspected race in the dataloader. Three weeks.

The cause: lifetimes were computed over the forward graph. Activations saved for backward were treated as dead after their last forward use, and their buffers were reused. The backward then read whatever had been written over them. Batch-size dependence is an allocation-pattern artifact; larger batches change which buffers collide.

The tell was structural and available from day one: inference conformance passed and training conformance was never run, because training conformance was assumed to be inference conformance plus an optimiser.

**GREEN.**

1. Lifetimes are computed over the **joint** forward+backward graph via AOTAutograd, so saved tensors carry their true lifetime.
2. `reuse_legal` gains the `needs_saving_for_backward` and alias conditions above.
3. Conformance runs gradient checks (`conformance-testing.md`), so a corrupted saved activation shows as a gradient mismatch immediately rather than as slow-motion training degradation.
4. Each reuse decision is recorded with the lifetime interval that justified it, so a wrong reuse can be found by reading the manifest instead of by bisection.

Generalisable: **any lifetime analysis over the forward graph alone is wrong for training.** The forward graph is a subgraph of what actually executes, and optimisations derived from a subgraph are valid only for programs that stop there. Verified earlier in this pack: for a two-layer model the backward graph has 18 nodes to the forward's 7 — most of the program is not in the forward.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Fusing a multi-user intermediate | Breaks residuals; the second consumer loses its input | `len(node.users) != 1` |
| Fusing over in-place ops | Memory deps are not graph edges | Functionalise first |
| Fused reduction accumulating in compute dtype | Plausible wrong numbers | Honour `accumulate_dtype` |
| Folding a trainable parameter | Silently freezes it | Consult the trainability mask |
| Folding based on current values being zero | Zero now ≠ constant | Mask, never contents |
| Folding pass that defaults to allow | Removes things nobody chose to remove | Default deny |
| Forward-only lifetime analysis | Saved-for-backward tensors get clobbered | Joint graph |
| Ignoring views/aliases in lifetimes | Intermittent, allocation-order-dependent corruption | Extend lifetime across aliases |
| Layout pass that changes logical shape | Not a layout pass | Reject |
| Treating layout differences as a tolerance issue | Same-kernel layouts have no noise floor; format changes must trace to a kernel choice | Exact for same-kernel; budget + manifest entry for kernel-changing formats |
| Optimisation fires without a manifest entry | Future differences un-diagnosable | Record at fire time |
| Manifest without a `condition` field | Cannot explain why A fused and B did not | Record shape/device/dtype |

---

## Checklist

- [ ] Fusion checks all five legality conditions, including `len(users) == 1`
- [ ] Exposed intermediates from the IO contract are never fused across
- [ ] In-place ops functionalised before any fusion pass
- [ ] Reassociating fusions require `contract.reassociation_allowed`
- [ ] Fused reductions accumulate in `accumulate_dtype`
- [ ] Layout passes never change logical shape
- [ ] Non-contiguous inputs exercised in conformance
- [ ] Constant folding consults a trainability mask and defaults to deny
- [ ] Lifetimes computed over the joint forward+backward graph
- [ ] Alias/view relationships extend lifetimes
- [ ] Graph outputs and saved-for-backward tensors excluded from reuse
- [ ] Every optimisation records kind, nodes before/after, legality basis, numerical impact, and condition
- [ ] Topology diff reconciles line-for-line against the manifest

---

## Related Sheets

- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — what fusion is permitted to change
- [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — reassociation permission and budgets
- [torch-fx-capture-and-transformation.md](torch-fx-capture-and-transformation.md) — functionalisation before optimisation
- [torch-compile-and-aotautograd.md](torch-compile-and-aotautograd.md) — obtaining the joint graph
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — the manifest these entries feed
