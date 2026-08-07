---
name: torch-compile-and-aotautograd
description: Use when driving torch.compile, dynamo, or inductor as a backend — guards and recompilation limits, graph breaks, custom backends — or when you need joint forward+backward capture via AOTAutograd. Also read when deciding whether eager-mode compilation is the right first implementation.
---

# torch.compile and AOTAutograd

## When to Use

- `torch.compile` is recompiling constantly, or is slower than eager.
- You need the *backward* graph, not just the forward — fusing across the backward, or hashing what you actually run.
- You are writing a custom backend and need to know where in the stack it plugs in.
- You are deciding whether to build your own lowering pipeline or lean on inductor.

---

## Core Principle

**`torch.compile` is a backend, not a contract. It gives you speed; it does not give you a program you can name, approve, or compare against. If your system needs semantic identity, you need your own IR *and* you can still use inductor underneath it.**

These are not alternatives. The productive arrangement is: your canonical IR carries identity, your passes do what only you can do, and inductor does codegen. What you must not do is let `torch.compile`'s opaque, guard-dependent, silently-re-derived artifact *be* your artifact — because you cannot hash it, cannot diff it, and cannot tell whether today's run used the same code as yesterday's.

### Start in eager mode

For a new lowering pipeline, the right first implementation is almost always **eager-mode compilation**: your passes produce a `GraphModule` that runs op-by-op in eager PyTorch. No codegen, no kernel authoring.

The reason is not caution, it is diagnosis. In eager mode a conformance failure localises to a node — you can print intermediates, bisect the graph, and compare per-node against the reference (`miscompile-taxonomy-and-debugging.md`). Once inductor generates a fused Triton kernel, the intermediates you want to inspect do not exist as tensors; a wrong answer is one wrong number at the end of a kernel you did not write. **Debuggability is a property you spend, and you should not spend it before conformance passes.**

Add inductor when eager-mode conformance is green and profiling shows codegen is what you need. The manifest records which backend produced the artifact, so the two are distinguishable forever after.

---

## The Stack

```
  your model / GraphModule
        │
        ▼
  ┌───────────┐  bytecode analysis; extracts FX graphs; installs GUARDS
  │  DYNAMO   │  falls back to Python on anything it cannot trace (graph break)
  └───────────┘
        │  FX graph (forward only)
        ▼
  ┌──────────────┐  traces the JOINT forward+backward, functionalises,
  │ AOTAUTOGRAD  │  applies decompositions, partitions fwd/bwd
  └──────────────┘
        │  two FX graphs (fw, bw), in ATen ops
        ▼
  ┌───────────┐  lowering + scheduling + Triton/C++ codegen
  │ INDUCTOR  │
  └───────────┘
```

The layer you care about depends on the job:

| Need | Layer |
|------|-------|
| A graph you own and can hash | `torch.fx` directly — see `torch-fx-capture-and-transformation.md` |
| Capture through Python control flow | dynamo |
| The backward graph | AOTAutograd |
| Kernel generation | inductor |

---

## Guards and Recompilation

Dynamo compiles a *specialised* version of your function and installs guards — runtime predicates that decide whether the cached code is valid. A guard failure means a recompile.

Verified on torch 2.9:

```python
import torch, torch._dynamo as dynamo
from torch._dynamo.utils import counters

def f(x, flag):
    return x * 2 if flag else x * 3

dynamo.reset(); counters.clear()
cf = torch.compile(f, dynamic=False)
for flag in (True, False, True):
    cf(torch.randn(4), flag)

print(dict(counters["frames"]))     # {'total': 2, 'ok': 2}
```

Two compilations for three calls: `flag=True` and `flag=False` are separate specialisations, and the third call reuses the first. A Python bool in the signature is a guard, and every distinct value is a distinct artifact.

What guards specialise on — each of these is a recompilation trigger:

| Guarded on | Typical trigger |
|-----------|-----------------|
| Tensor dtype, device, requires_grad | Mixed precision, moving to GPU mid-run |
| Tensor rank, and shape when `dynamic=False` | Variable batch size, last partial batch |
| Python scalar *values* (bool, int) in the signature | Config flags, step counters |
| `id()` of some objects, module attributes | Mutating `self.config` between calls |
| Global state (grad mode, autocast) | `model.train()` / `model.eval()` |

The default recompilation limit is 8 (`torch._dynamo.config.recompile_limit`, also exposed as `cache_size_limit`). Exceed it and dynamo **falls back to eager for that frame** — quietly. The symptom is a model that was fast for an hour and then is not, with no error.

Diagnosis:

```python
import torch._dynamo as dynamo
torch._logging.set_logs(recompiles=True)     # prints the guard that failed, and why
```

Fixes, in order of preference: make the varying thing a tensor rather than a Python scalar; pass `dynamic=True` to compile a shape-polymorphic version; mark specific dimensions with `torch._dynamo.mark_dynamic(x, 0)`; only as a last resort raise the limit — a high limit usually means you are compiling many artifacts and benefiting from none.

**For a compiler pipeline, guard behaviour is a correctness concern, not just performance.** A guard that fails silently into eager means the artifact you conformance-checked is not the code that ran. If you use `torch.compile` inside a pipeline that makes identity claims, either compile with `fullgraph=True` (raise instead of breaking) or record the compiled-versus-eager fallback state in the manifest.

---

## Graph Breaks

When dynamo cannot trace something, it splits the region and runs the untraceable part in Python:

```python
import torch._dynamo as dynamo

def g(x):
    y = x * 2
    print("side effect")     # untraceable
    return y + 1

exp = dynamo.explain(g)(torch.randn(3))
print(exp.graph_count)       # 2
print(exp.break_reasons[0].reason)
# "Failed to trace builtin operator / Explanation: Dynamo does not know how to trace builtin ..."
```

Two graphs where you expected one. Common causes: printing and logging, `.item()` / `.tolist()` / `numpy()` conversions, data-dependent control flow, custom autograd functions dynamo has not been taught, and exception handling around traced code.

`dynamo.explain` is the tool. Run it before optimising anything — a fusion that cannot cross a graph break is a fusion that will not happen, and a "why is my fused kernel not firing" investigation usually terminates at a `print` left in from debugging.

For a compiler that must capture the whole program, use `fullgraph=True` so a break is an error rather than a silent partial capture.

---

## AOTAutograd: Getting the Backward Graph

Dynamo gives you the forward. Most of the correctness risk lives in the backward. AOTAutograd traces the joint forward-and-backward and hands you both:

```python
import torch, torch.nn as nn
from torch._functorch.aot_autograd import aot_module_simplified

captured = {}

def fw_compiler(gm, example_inputs):
    captured["fw"] = gm            # forward graph, ATen ops, functionalised
    return gm.forward

def bw_compiler(gm, example_inputs):
    captured["bw"] = gm            # backward graph — the thing forward-only testing misses
    return gm.forward

def backend(gm, example_inputs):
    return aot_module_simplified(gm, example_inputs,
                                 fw_compiler=fw_compiler, bw_compiler=bw_compiler)

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
compiled = torch.compile(model, backend=backend)
compiled(torch.randn(2, 4, requires_grad=True)).sum().backward()

print(len(captured["fw"].graph.nodes), len(captured["bw"].graph.nodes))   # 7 18
```

The backward graph has 18 nodes to the forward's 7 — verified on torch 2.9 for a two-layer model. **The backward is the larger program.** That ratio is the quantitative argument for gradient conformance: forward-only testing exercises under a third of what the artifact actually runs.

What AOTAutograd does for you that is hard to do yourself:

- **Functionalisation** — in-place ops become out-of-place, so memory dependencies become graph edges (the hazard in `torch-fx-capture-and-transformation.md`).
- **Decomposition** — composite ops lower to a smaller ATen core set. `torch._decomp.core_aten_decompositions()` is 1014 entries on torch 2.9.1; consuming that table is far cheaper than writing your own, and it means your backward is derived by the same rules PyTorch uses. See `operator-lowering-and-kernel-selection.md`.
- **Partitioning** — decides what the forward saves versus what the backward recomputes (the activation-checkpointing tradeoff).

Note the API surface: `aot_module_simplified`, `aot_module`, and `aot_function` all live under `torch._functorch.aot_autograd` — a **private** module. It is what `torch.compile` itself uses and it is stable in practice, but it is not covered by API-stability guarantees. Pin your torch version and record it in the manifest (`compilation-manifests-and-reproducibility.md`); a minor-version bump can change the joint graph you capture, which changes your artifact.

---

## Executable Decision Procedure: Which Capture Mechanism?

```python
def choose_capture(*, need_semantic_hash: bool, need_backward_graph: bool,
                   has_data_dependent_control_flow: bool, need_codegen: bool,
                   conformance_gate_exists: bool) -> tuple[str, str]:
    if need_semantic_hash and not conformance_gate_exists:
        return ("STOP", "Build the conformance gate before choosing a backend. "
                        "Otherwise the backend's behaviour becomes your specification. "
                        "See conformance-testing.md")
    if need_semantic_hash:
        base = ("torch.fx (own the graph)",
                "You need a graph you can hash, diff, and approve. Trace with fx; "
                "run your passes; hash the result. ")
        if need_backward_graph:
            base = (base[0] + " + AOTAutograd",
                    base[1] + "Add aot_module_simplified to capture the joint graph, "
                    "and hash the JOINT graph — the backward is the larger program. ")
        if need_codegen:
            base = (base[0] + " + inductor backend",
                    base[1] + "Use inductor for codegen only, AFTER eager-mode "
                    "conformance is green. Record the backend in the manifest. ")
        if has_data_dependent_control_flow:
            base = (base[0], base[1] + "fx cannot trace data-dependent control flow: "
                    "use torch.cond / capture per-branch subgraphs, or accept that "
                    "each branch is a separate artifact with its own identity.")
        return base
    if has_data_dependent_control_flow:
        return ("torch.compile(fullgraph=False)",
                "No identity requirement and dynamic control flow: let dynamo break "
                "graphs. Check dynamo.explain() so you know where the breaks are.")
    return ("torch.compile(fullgraph=True)",
            "No identity requirement, static control flow: simplest thing that works. "
            "fullgraph=True so a silent partial capture becomes an error.")
```

The `STOP` branch is the one that matters. Choosing a backend before a gate exists means the first thing the gate ever sees is whatever inductor produced, and "matches inductor" quietly becomes the definition of correct.

---

## RED → GREEN Scenario

**RED.** A team wraps their training step in `torch.compile` and reports a 1.8× speedup from a microbenchmark. In production the speedup is 1.05×. Investigation over two weeks blames the dataloader.

The actual causes, all visible in thirty seconds with the right tool:

1. `dynamo.explain` shows `graph_count=4` — a `print` in the loss function and a `.item()` for logging split the region into four graphs, so nothing fuses across them.
2. `torch._logging.set_logs(recompiles=True)` shows recompilation on every last-partial-batch, because `dynamic=False` guards on exact shape.
3. After 8 recompiles the frame exceeds `recompile_limit` and falls back to eager permanently — the "compiled" model is running eager for most of the epoch.

The microbenchmark used a fixed batch size and no logging, so it measured none of this.

**GREEN.**

```python
import torch, torch._dynamo as dynamo

exp = dynamo.explain(train_step)(batch)
assert exp.graph_count == 1, [str(r.reason) for r in exp.break_reasons]

torch._logging.set_logs(recompiles=True)
compiled = torch.compile(train_step, dynamic=True, fullgraph=True)
```

`fullgraph=True` turns the `print` from a silent 4× fragmentation into an immediate error. `dynamic=True` compiles one shape-polymorphic artifact instead of one per batch size. Recompile logging makes guard failures visible rather than inferred.

And the benchmark is fixed to match production: variable batch sizes, logging enabled, a full epoch. The generalisable point — **a microbenchmark that omits the conditions that trigger guards measures a program that will never run.** Guard behaviour is a property of the workload, not of the model.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| `torch.compile` output treated as the artifact | Opaque, guard-dependent, unhashable | Own an IR; use inductor for codegen |
| Codegen before eager-mode conformance | Spends debuggability before it is earned | Eager first; add inductor when green |
| `fullgraph=False` in an identity-bearing pipeline | Silent partial capture | `fullgraph=True` |
| Ignoring the recompile limit | Silent permanent fallback to eager | Log recompiles; `dynamic=True` |
| Python scalars in a compiled signature | Every value is a new artifact | Pass tensors |
| Benchmarking with fixed shapes and no logging | Measures a program that never runs | Benchmark the real workload |
| Forward-only conformance on a compiled model | The backward is the larger graph | Capture and check it — `conformance-testing.md` |
| Unpinned torch with `_functorch` APIs | Private API; joint graph can change between versions | Pin and record in the manifest |

---

## Checklist

- [ ] Eager-mode conformance is green before inductor is enabled
- [ ] `dynamo.explain()` run; `graph_count` is understood and asserted
- [ ] `fullgraph=True` wherever identity claims are made
- [ ] `torch._logging.set_logs(recompiles=True)` used during bring-up
- [ ] No Python scalars in compiled signatures that vary at runtime
- [ ] `dynamic=True` or `mark_dynamic` where shapes vary
- [ ] Backward graph captured (AOTAutograd) and included in conformance
- [ ] torch version pinned and recorded; `_functorch` usage flagged as private-API dependence
- [ ] Backend choice (eager / inductor / custom) recorded in the manifest
- [ ] Benchmarks reproduce production shape and logging conditions

---

## Related Sheets

- [torch-fx-capture-and-transformation.md](torch-fx-capture-and-transformation.md) — the graph you own
- [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — the decomposition table AOTAutograd applies
- [conformance-testing.md](conformance-testing.md) — why the backward graph must be checked
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — recording backend and versions
- [cost-estimation-and-compilation-budgets.md](cost-estimation-and-compilation-budgets.md) — whether compile time pays back
