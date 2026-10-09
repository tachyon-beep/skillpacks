---
name: torch-fx-capture-and-transformation
description: "Use when capturing a model or IR into torch.fx and transforming it \u2014 symbolic tracing, fx.Graph surgery, writing and composing passes, GraphModule round-trips \u2014 and when tracing fails or misbehaves on control flow, dynamic shapes, or in-place operations."
---

# torch.fx Capture and Transformation

## When to Use

- You need a graph you can walk, rewrite, and hash. `torch.fx` is usually the cheapest way to get one.
- `symbolic_trace` throws, or produces a graph that is subtly not the program.
- You are writing more than one pass and they are starting to interfere.
- A rewritten `GraphModule` runs but produces wrong numbers.

For dynamo/inductor (a different capture mechanism with different tradeoffs) see `torch-compile-and-aotautograd.md`. The short version: **fx gives you a graph you own; dynamo gives you a fast artifact you mostly do not.** For building a compiler, you usually want the graph you own.

---

## Core Principle

**`symbolic_trace` records the ops that a *single* symbolic execution reached. Anything Python decided at trace time is gone — it is not in the graph, and its absence is silent.**

That is the whole class of fx bugs. The graph is not a translation of the source; it is a transcript of one walk through it.

### The failure this prevents

```python
class Model(nn.Module):
    def forward(self, x, training_mode=True):
        if training_mode:
            x = self.dropout(x)
        return self.head(x)

gm = fx.symbolic_trace(Model())   # traced with training_mode=True (the default)
```

The graph contains dropout unconditionally. `training_mode=False` is now unrepresentable: the compiled artifact applies dropout in evaluation. No error, no warning — the `if` was a Python decision, resolved at trace time, and the branch not taken left no trace.

This is worse than a crash because it produces a plausible artifact. The eval-mode numbers are slightly wrong in a way that looks like regularisation noise.

`symbolic_trace` *will* raise when the condition depends on a traced tensor:

```python
class DD(nn.Module):
    def forward(self, x):
        if x.sum() > 0: return x * 2
        return x * 3

fx.symbolic_trace(DD())
# TraceError: symbolically traced variables cannot be used as inputs to control flow
```

That error is a gift. The dangerous case is the silent one, where the condition depends on a Python argument, a config value, or `self.something` — tracing succeeds and bakes in one branch.

**Rule: after tracing, read `gm.code` and confirm it is the program you meant.** It takes ten seconds and it is the only check that catches trace-time specialisation.

---

## What the Graph Contains

Five node kinds, and the distinction between them drives every pass you write:

| `node.op` | `node.target` | Notes |
|-----------|---------------|-------|
| `placeholder` | arg name | Graph inputs |
| `get_attr` | attribute path | Parameters/buffers — *state*, not semantics. Exclude from the semantic hash |
| `call_function` | the function object | `torch.relu`, `operator.add` |
| `call_method` | method name (str) | `x.view(...)`, `x.sum(...)` |
| `call_module` | submodule path (str) | `self.lin` — opaque; fx does not look inside |
| `output` | — | Exactly one |

`call_module` being opaque matters. `fx.symbolic_trace` does not descend into leaf modules, so `nn.Linear` is one node, not `matmul + add`. If your passes need to see inside, either trace with a custom `Tracer` whose `is_leaf_module` returns `False`, or run a decomposition first (`operator-lowering-and-kernel-selection.md`).

Shape and dtype are **not** in the graph by default. To get them:

```python
from torch.fx.passes.shape_prop import ShapeProp
ShapeProp(gm).propagate(example_input)
# node.meta['tensor_meta'].shape / .dtype now populated
```

Verified on torch 2.9: after `ShapeProp`, each node carries `meta['tensor_meta']` with concrete shape and dtype. Passes that make layout or fusion decisions need this; running them without it means deciding on assumptions.

---

## Graph Surgery That Works

The four-line idiom for replacing a node, with the ordering that matters:

```python
import torch, torch.fx as fx

for node in list(gm.graph.nodes):            # list() — you are mutating while iterating
    if node.op == "call_function" and node.target is torch.relu:
        with gm.graph.inserting_after(node):
            new = gm.graph.call_function(torch.nn.functional.silu, node.args, node.kwargs)
        node.replace_all_uses_with(new)      # rewire consumers BEFORE erasing
        gm.graph.erase_node(node)            # now it has no users

gm.graph.lint()      # topological order, no dangling refs, single output
gm.recompile()       # regenerate forward() — without this you run the OLD code
```

Four failure modes, all common:

- **Iterating `gm.graph.nodes` while mutating** — the iterator is over a live linked list. Snapshot with `list()`.
- **`erase_node` before `replace_all_uses_with`** — raises, because the node still has users. If you see that error, your order is backwards.
- **Forgetting `recompile()`** — the graph is correct, `gm.forward` is stale, and the artifact runs the pre-pass program. Every conformance check passes against the *old* code and the bug appears only after some unrelated change forces a recompile.
- **Skipping `lint()`** — a pass that inserts a node in the wrong position produces a graph that is fine until a later pass reorders something.

`inserting_after(node)` rather than `inserting_before(user)` is deliberate: it keeps the new node adjacent to what it replaced, so the topology diff in `ir-contracts-and-semantic-identity.md` stays readable.

---

## Composing Passes

Passes interfere. Two that are individually correct can be wrong in sequence — a fusion pass that assumes shapes are annotated, run after a pass that inserted un-annotated nodes, will make decisions on missing metadata.

Give each pass an explicit contract and enforce it:

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Pass:
    name: str
    run: Callable                       # (GraphModule) -> GraphModule
    requires: frozenset = frozenset()   # e.g. {"shapes", "decomposed"}
    provides: frozenset = frozenset()
    invalidates: frozenset = frozenset()
    may_change_topology: bool = False    # see ir-contracts-and-semantic-identity.md

def run_pipeline(gm, passes: list[Pass], manifest: list) -> "GraphModule":
    state: set[str] = set()
    for p in passes:
        missing = p.requires - state
        if missing:
            raise PipelineError(f"pass {p.name!r} requires {sorted(missing)}; "
                                f"available: {sorted(state)}")
        before = semantic_skeleton(gm)
        gm = p.run(gm)
        gm.graph.lint()
        gm.recompile()

        changed = topology_delta(before, semantic_skeleton(gm))
        if changed and not p.may_change_topology:
            raise ContractViolation(f"pass {p.name!r} changed topology: {changed}")

        manifest.append({"pass": p.name, "topology_delta": changed})   # ALWAYS append
        state = (state - p.invalidates) | p.provides
    return gm
```

Three properties this buys, each mapping to a failure it prevents:

- **`requires`/`provides`** — a pass cannot silently run on stale metadata. Reordering the pipeline fails loudly instead of producing wrong decisions.
- **`may_change_topology`** — the permission from the IR contract, enforced per pass rather than assumed. A pass that was not supposed to add nodes and did is caught at the pass, not five stages later at the gate.
- **Unconditional manifest append** — including for passes that changed nothing. A pass missing from the manifest is indistinguishable from a pass that did not run; recording no-ops keeps the manifest a complete record. See `compilation-manifests-and-reproducibility.md`.

---

## Three Pitfalls

### In-place operations

In-place ops trace successfully and then break passes:

```python
class M(nn.Module):
    def forward(self, x):
        h = self.l(x)
        h.relu_()          # traces to a node named 'relu_'
        return h + 1.0
```

The graph records `relu_` as a node whose *return value* is unused by `add` — `add` consumes `l` directly. The dataflow edges no longer describe the dependency, because the dependency is through memory, not through the graph. Consequences:

- Dead-code elimination sees `relu_` as unused and removes it. The program changes.
- Reordering passes may move `add` before `relu_`.
- Buffer reuse in memory planning can alias a still-live tensor.

**Functionalise before optimising.** Convert in-place ops to their out-of-place equivalents so memory dependencies become graph edges, then let the memory planner reintroduce in-place where it can prove safety (`fusion-and-memory-planning.md`). Detect them first:

```python
INPLACE = lambda n: (n.op == "call_method" and isinstance(n.target, str)
                     and n.target.endswith("_") and not n.target.startswith("__"))

inplace_nodes = [n.name for n in gm.graph.nodes if INPLACE(n)]
if inplace_nodes:
    raise PipelineError(f"functionalise before optimising: {inplace_nodes}")
```

### Dynamic shapes

`symbolic_trace` handles shape *access* symbolically — `x.shape[0]` becomes `getattr` + `getitem` nodes, and `x.numel()` becomes a `call_method` node, both verified on torch 2.9. What it cannot handle is shape *arithmetic that Python evaluates*:

```python
def forward(self, x):
    return x.view(x.shape[0], -1)      # fine — shape access is traced

def forward(self, x):
    n = int(x.shape[0])                 # int() forces a Python value
    return x.view(n // 2, -1)           # n is BAKED IN
```

The second bakes the traced batch size into the graph. It runs, and is wrong for every other batch size — often producing a shape error far downstream, occasionally producing a silently wrong reshape.

**Check for baked constants after tracing:** search `gm.code` for integer literals matching your example input's dimensions. If the example shape appears as a literal, it was specialised. Either trace with a shape the literal cannot coincide with, or record the specialisation in the IO contract so the artifact declares the batch sizes it is valid for.

### `call_module` opacity

A pass looking for `torch.matmul` will not find the one inside `nn.Linear`. Either decompose first, or trace with:

```python
class FlatTracer(fx.Tracer):
    def is_leaf_module(self, m, qualname): return False   # descend into everything
```

Descending gives more optimisation surface and a larger, slower-to-transform graph. Decide per pipeline, and record which you chose in the manifest — two artifacts built with different tracer configurations are different artifacts.

---

## Executable Decision Procedure: Post-Trace Validation

Run immediately after `symbolic_trace`, before any pass. Every check corresponds to a silent-wrongness mode above.

```python
def validate_trace(gm, example_inputs, declared_shapes, original_module) -> list[str]:
    findings = []

    # 1. Trace-time branch specialisation
    src = gm.code
    for branch_kw in ("if ", "for ", "while "):
        if branch_kw in src:
            findings.append(f"control flow survived tracing ({branch_kw.strip()!r}) — "
                            "unexpected; inspect gm.code")

    # 2. Shape literals baked in
    for dim in {d for shape in declared_shapes for d in shape if d > 1}:
        if f"{dim}," in src or f"({dim})" in src:
            findings.append(f"literal {dim} appears in traced code — possible shape "
                            "specialisation; verify or declare in the IO contract")

    # 3. In-place ops
    ip = [n.name for n in gm.graph.nodes
          if n.op == "call_method" and isinstance(n.target, str)
          and n.target.endswith("_") and not n.target.startswith("__")]
    if ip:
        findings.append(f"in-place ops present, functionalise before optimising: {ip}")

    # 4. Behavioural equivalence to the thing you traced — forward AND backward
    with torch.no_grad():
        if not torch.allclose(gm(*example_inputs), original_module(*example_inputs),
                              rtol=1e-6, atol=1e-7):
            findings.append("traced module does not match source module — "
                            "tracing dropped or specialised something")

    # 5. Structural sanity
    try:
        gm.graph.lint()
    except Exception as e:
        findings.append(f"graph.lint() failed: {e}")

    return findings
```

Check 4 is the one that earns its keep. It catches every trace-time specialisation *whose effect is visible at the example input* — which is most of them, and it costs one forward pass. Run it in both train and eval mode, because that is exactly the axis the dropout example above breaks on.

---

## RED → GREEN Scenario

**RED.** A team writes an operator-substitution pass that replaces `torch.relu` with a fused activation. They test it:

```python
gm = fx.symbolic_trace(model)
apply_activation_fusion(gm)
assert torch.allclose(gm(x), model(x))    # passes
```

It passes because they forgot `gm.recompile()` — `gm.forward` is still the pre-pass code, so the assertion compares the original model against itself. The pass is never exercised. Three weeks later an unrelated change adds a second pass that *does* call `recompile()`, the fusion pass's output finally executes, and a shape bug that has been in it since day one surfaces — in a release build, attributed to the innocent second pass.

**GREEN.** The pipeline runner calls `lint()` and `recompile()` after every pass (as in `run_pipeline` above), and the pass test asserts the pass actually fired:

```python
def test_activation_fusion():
    gm = fx.symbolic_trace(model)
    before = [n.target for n in gm.graph.nodes]
    manifest = []
    gm = run_pipeline(gm, [ACTIVATION_FUSION_PASS], manifest)

    assert manifest[0]["topology_delta"], "pass did not fire — test proves nothing"
    assert torch.relu not in [n.target for n in gm.graph.nodes]
    assert torch.allclose(gm(x), model(x), rtol=1e-6, atol=1e-7)
```

The first assertion is the load-bearing one. **A transformation test that does not verify the transformation occurred is a test of the identity function** — and it passes forever, which is why nobody notices.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Trusting a trace without reading `gm.code` | Trace-time branch specialisation is silent | Read it; validate in both modes |
| Iterating `gm.graph.nodes` while mutating | Live linked list | `list(gm.graph.nodes)` |
| `erase_node` before `replace_all_uses_with` | Raises, or orphans users | Rewire, then erase |
| No `recompile()` | Runs stale code; tests compare a module to itself | Recompile in the pipeline runner, not per pass |
| No `lint()` | Malformed graphs surface passes later | Lint after every pass |
| Optimising over in-place ops | Memory deps are not graph edges | Functionalise first |
| `int(x.shape[0])` | Bakes the traced batch size | Keep shape arithmetic symbolic, or declare it |
| Passes with implicit ordering | Stale metadata used silently | `requires`/`provides` contracts |
| Manifest entry only when a pass changes something | No-op and did-not-run are indistinguishable | Append unconditionally |
| Transformation test with no "did it fire" assertion | Tests the identity function | Assert the topology delta |

---

## Checklist

- [ ] `gm.code` read after tracing, in both train and eval mode
- [ ] Traced module verified against the source module numerically
- [ ] `ShapeProp` run before any layout- or fusion-dependent pass
- [ ] In-place ops detected and functionalised before optimisation
- [ ] Shape literals checked against declared shapes
- [ ] Pipeline runner calls `lint()` and `recompile()` after every pass
- [ ] Passes declare `requires` / `provides` / `may_change_topology`
- [ ] Topology delta recorded per pass, unconditionally
- [ ] Each pass test asserts the pass fired
- [ ] Tracer leaf-module configuration recorded in the manifest

---

## Related Sheets

- [torch-compile-and-aotautograd.md](torch-compile-and-aotautograd.md) — the other capture mechanism, and when to prefer it
- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — the topology-delta check used above
- [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — decomposing `call_module` opacity away
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — why functionalisation comes first
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — what each pass records
