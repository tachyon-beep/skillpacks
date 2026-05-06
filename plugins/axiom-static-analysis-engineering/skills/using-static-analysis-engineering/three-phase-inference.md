---
name: three-phase-inference
description: Use when designing the inference algorithm of a dataflow analyzer — how lattice values propagate from variable assignments, through function bodies, across the callgraph, until a fixed point is reached. Covers the variable → function → callgraph phasing rationale, the worklist algorithm, termination proofs grounded in lattice properties, cycle handling (recursion, mutually recursive callgraphs), and the whole-program vs incremental tradeoff. Produces `03-inference-pipeline-spec.md`.
---

# Three-Phase Inference

## Why Phase the Inference

A dataflow analyzer that tries to do everything at once — solve variable values, function summaries, and callgraph propagation in a single fixed-point — is harder to reason about, harder to debug, harder to make incremental, and almost always slower than the same analyzer solved in phases.

The standard phasing for a typed dataflow engine:

| Phase | Inputs | Output | Domain |
|-------|--------|--------|--------|
| **1. Variable** | AST + lattice (`02-`) + transfer functions | per-program-point map: `Variable → LatticeValue` within each function body | **Intra-procedural** |
| **2. Function** | Phase 1 outputs + parameter/return positions | per-function summary: `(input lattice values) → output lattice value` | **Function-local** |
| **3. Callgraph** | Phase 2 summaries + callgraph (who calls whom) | per-callsite lattice value, propagated across functions | **Inter-procedural** |

Each phase consumes only the previous one. The entire engine is three nested fixed-point loops, each with a tight termination argument.

`03-inference-pipeline-spec.md` is where you commit to this phasing, justify any deviation, and write the termination proof.

## Phase 1: Variable Inference (Intra-Procedural Worklist)

For each function body, compute the lattice value at every program point.

```python
def infer_function_body(body: Block, initial_env: Env) -> dict[ProgramPoint, Env]:
    """
    Worklist algorithm. Env is a mapping Variable -> LatticeValue.
    Block is the CFG (basic blocks + edges) for the function body.
    """
    env_in: dict[ProgramPoint, Env] = {pp: bottom_env() for pp in body.points}
    env_out: dict[ProgramPoint, Env] = {pp: bottom_env() for pp in body.points}
    env_in[body.entry] = initial_env

    worklist: deque[ProgramPoint] = deque([body.entry])
    while worklist:
        pp = worklist.popleft()
        new_out = transfer(pp, env_in[pp])  # apply the transfer function
        if new_out != env_out[pp]:
            env_out[pp] = new_out
            for succ in body.successors(pp):
                merged = join_env(env_in[succ], new_out)
                if merged != env_in[succ]:
                    env_in[succ] = merged
                    worklist.append(succ)
    return env_in
```

**Termination requires:**

- Lattice has finite ascending chains (or widening at depth $k$).
- Transfer functions are monotonic.
- `join_env` is the pointwise lattice join.

These properties come straight from `02-`. If `02-` doesn't establish them, this loop doesn't terminate; revisit `02-` rather than papering with iteration caps.

**Common subtleties:**

- **CFG construction** — branches, exception edges, finally blocks, generator yield points. Missing an edge silently shrinks the analysis. Build the CFG once, audit it, reuse it.
- **Loops** — a loop is a cycle in the CFG. The fixed-point handles loops naturally if (and only if) the lattice has finite height. Loops over infinite-height domains (e.g., interval-of-int) require widening per iteration.
- **Phi nodes / SSA** — if you choose SSA for the IR, joins happen at phi nodes; if you stay on the AST, joins happen at CFG merge points. Both work; SSA simplifies later phases at the cost of an SSA construction pass.
- **Dead code** — unreachable basic blocks should never be visited; they pollute the env with $\bot$ that may join into reachable code if the CFG is wrong.

## Phase 2: Function Inference (Summary Construction)

Given Phase 1's per-point environments, build a **function summary**: a relation between the lattice values at the function's *entry* (parameters, captured variables, globals it reads) and at its *exit* (return value, side effects on globals it writes).

```python
@dataclass
class FunctionSummary:
    fn: Function
    parameter_effects: dict[ParamIndex, Callable[[LatticeValue], LatticeValue]]
    return_value: Callable[[tuple[LatticeValue, ...]], LatticeValue]  # over all params
    global_writes: dict[GlobalName, LatticeValue]
    sinks_reached: list[SinkSite]  # lattice value at each sink in this function
```

**Why summarise:**

- Phase 3 propagates across the callgraph. Without summaries, every callsite re-analyses the callee body, which is exponential in call depth.
- Summaries make the engine **incremental**: if a function body doesn't change, its summary doesn't change, and only callers need re-propagation.
- Summaries are the natural unit for **library stubs** (a third-party function whose body is unavailable but whose summary is hand-written; `cross-module-flow-analysis.md` covers this in depth).

**Common subtleties:**

- **Polymorphism in the lattice** — a function whose return tier depends on its argument tier (e.g., `def identity(x): return x` returns whatever was passed) requires the summary to be a *function* over input tiers, not a fixed output tier. Most engines special-case identity-like functions; better to make summaries first-class polymorphic.
- **Side effects on globals** — Python's module-level mutable state, globals modified by callees, captures in closures. The summary must record these or Phase 3 sees stale data. The simplest sound option: any function that touches a global $\sqcup$ s the global with $\top$ in its summary; refine later.
- **Pure functions** — if a function reads no globals and writes no globals, the summary is just `(params) → return`. Note them; many real codebases have ~70% pure functions, and noting purity speeds Phase 3 dramatically.
- **Generators and coroutines** — yield interleaves the function with its caller. Either model the function as a state machine (sound but heavy) or treat each yield as a sink/source pair (lossy but tractable).

## Phase 3: Callgraph Propagation

Given Phase 2 summaries and a callgraph (a graph whose nodes are functions, whose edges are call relationships), propagate lattice values across function boundaries until the callgraph itself reaches a fixed point.

```python
def propagate_callgraph(callgraph: CallGraph, summaries: dict[Function, FunctionSummary]) -> dict[CallSite, Env]:
    """
    For each callsite (caller, callee, call_args), use the callee's summary to
    compute the env after the call. Iterate until callsite envs stabilise.
    """
    callsite_env: dict[CallSite, Env] = {cs: bottom_env() for cs in callgraph.sites}
    worklist: deque[CallSite] = deque(callgraph.sites)
    while worklist:
        cs = worklist.popleft()
        summary = summaries[cs.callee]
        new_env = apply_summary(summary, cs.caller_env_at_call)
        if new_env != callsite_env[cs]:
            callsite_env[cs] = new_env
            # propagating may invalidate callers' summaries
            for invalidated in callgraph.callers(cs.caller):
                worklist.append(invalidated)
    return callsite_env
```

**The recursion problem:** mutually recursive functions appear in the callgraph as a cycle. Naïve iteration would re-analyse forever. Standard solutions:

- **SCC condensation** — collapse strongly connected components into super-nodes; analyse each SCC together until its summaries stabilise; then move to the next SCC in topological order. This is the standard textbook approach.
- **Worklist with summary-equality check** — same as the variable phase, but at the function-summary level. Terminates iff summaries are in a finite-height domain (which they are, if the underlying lattice from `02-` is).
- **Bottom-up summarisation** — analyse functions in reverse topological order; for SCCs, iterate to fixed point within the SCC. This is what most production analyzers do.

**Common subtleties:**

- **Callgraph completeness** — virtual dispatch, dynamic imports, `getattr`, `eval`, decorator-modified callables. If the callgraph is missing an edge, propagation misses flow. The conservative move is to treat unknown callees as $\top$ (returns worst-case, writes worst-case to all globals); the precise move requires deeper resolution. `callgraph-construction.md` is dedicated to this.
- **Higher-order functions** — `map(f, xs)` is a callsite where `f` depends on the value of an argument. Either specialise per known `f` (more precise; combinatorially explosive) or treat `f` as a value of "callable lattice" with a summary that is the join over possible callees (sound; less precise).
- **Exceptions** — a callee that raises propagates control back to the caller's exception handler. Many analyzers ignore this and silently miss flow through `except` clauses. State the choice in `03-`.
- **Library stubs** — for callees whose bodies are unavailable (stdlib, third-party), use hand-written summaries. The summary discipline is the same as Phase 2; the source is human, not the engine.

## Termination Proof Template

`03-` must contain (at this level of explicitness):

```
TERMINATION

Phase 1 (variable inference) terminates because:
  (a) The lattice from `02-` has finite ascending chains
      [or: widening operator W applied at depth k bounds chain length to k+1].
  (b) Transfer functions are monotonic [proven in `02-` per-operation].
  (c) The CFG is finite (the function body is finite).
  (d) The worklist algorithm strictly increases env values per visit;
      by (a)+(b), each program point can be revisited at most height(L) times.
  Bound: O(|CFG| × height(L)) work per function.

Phase 2 (function summary construction) terminates because:
  (a) Each function summary is a finite tuple of lattice values [from `02-`].
  (b) Summaries are joined monotonically as Phase 1 reaches more points.
  (c) Each function is summarised exactly once per Phase 1 fixed-point;
      summary construction is bounded by the summary domain's height.

Phase 3 (callgraph propagation) terminates because:
  (a) The callgraph has finitely many callsites (program is finite).
  (b) Callsite envs live in the same finite-height domain (lifted to tuples).
  (c) Per (a)+(b), each callsite can be revisited at most height(L)^arity(callee) times.
  (d) Recursive cycles are handled by SCC condensation
      [or: by the same monotonicity argument applied at the summary level].
  Bound: O(|CallSites| × height(L)^k) for some k bounded by max arity.
```

If any line of this proof is hand-waved, inference may not terminate, or may terminate at the wrong fixed point.

## Whole-Program vs Incremental

Two operating modes the engine should choose between (or support both):

**Whole-program** — re-analyse the entire codebase from scratch. Conceptually simple. Acceptable for codebases under ~100k LoC and analysers that take seconds. Above that, becomes the bottleneck.

**Incremental** — given a change set (a set of modified files), re-analyse only:

1. Phase 1 for every changed function body.
2. Phase 2 for every function whose body changed (summary may have changed).
3. Phase 3 for every callsite that calls a function whose summary changed, *and* every transitive caller of those (since their summaries may also have changed).

Incremental analysis requires:

- **Cached Phase 2 summaries** keyed by (function-body-hash, lattice-version, analyzer-version).
- **A reverse callgraph** to identify callers efficiently.
- **A Phase 3 invalidation rule**: a summary change invalidates callsites of the changed function, which may invalidate the callers' summaries, which may invalidate *their* callsites, and so on. Run the invalidation to a fixed point before re-running Phase 3.

The cache is the hard part. Stale cache hits are silent wrong answers. Cache key must include lattice version (`02-` semver) and analyzer version (`99-` semver), or every lattice change requires a full cache flush.

`scaling-to-large-codebases.md` covers the full incremental story; this sheet covers the architecture.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| All three phases interleaved into one big fixed-point | Hard to debug, hard to make incremental, often slower | Separate the phases; each is a clean worklist |
| Skipping Phase 2 (no function summaries) | Exponential cost in call depth | Build summaries; cache them |
| Callgraph missing edges (virtual dispatch, dynamic imports) | False negatives | Conservatively treat unresolved callees as $\top$; refine via `callgraph-construction.md` |
| No SCC handling for recursion | Phase 3 doesn't terminate | Condense SCCs; iterate within each |
| Library calls treated as identity | Massive false negatives at framework boundaries | Hand-written summaries (stubs); see `cross-module-flow-analysis.md` |
| Cache invalidation misses lattice version | Stale cache yields wrong findings after `02-` change | Cache key includes lattice and analyzer version |
| No termination proof | Inference loops on certain inputs | Write the proof template; if it doesn't compose, fix `02-` |
| Higher-order calls specialised exhaustively | Combinatorial explosion | Either specialise + iteration cap or use callable-lattice with joined summary |

## The Decision Output (`03-inference-pipeline-spec.md`)

A complete `03-` answers:

1. **Phase definitions** — the three phases, their inputs, their outputs, their domains.
2. **CFG construction** — what counts as an edge (control flow, exceptions, generators), how phi nodes / merges are handled.
3. **Worklist algorithm** — explicit pseudocode for each phase; queue ordering (FIFO, priority, postorder).
4. **Summary representation** — `FunctionSummary` schema; what's polymorphic, what's monomorphic, what's pure.
5. **Callgraph construction reference** — pointer to `callgraph-construction.md` for resolution depth.
6. **Recursion / cycle handling** — SCC condensation? Summary-level worklist? Iteration bound.
7. **Higher-order handling** — specialise, callable-lattice, or hybrid.
8. **Exception handling** — does the engine track `except` flow? Stated explicitly.
9. **Termination proof** — the template above, fully filled in.
10. **Whole-program vs incremental** — which the engine supports; cache key composition; invalidation rule.
11. **Soundness boundary** — what flows the engine intentionally doesn't see (eval, dynamic dispatch beyond resolution depth, threads, IPC).

## Cross-References

- `ast-visitation-patterns.md` — feeds the AST + CFG that Phase 1 walks
- `taint-lattice-design.md` — provides the lattice whose properties Phase 1's termination relies on
- `plugin-architecture-for-analyzer-rules.md` — rules consume Phase 3's per-callsite envs (sink reached with what tier)
- `false-positive-economics.md` — over-approximation in any phase shows up as FPs at the rule layer
- `callgraph-construction.md` — full treatment of resolution, virtual dispatch, dynamic imports
- `cross-module-flow-analysis.md` — Phase 3 across module boundaries with library stubs
- `scaling-to-large-codebases.md` — incremental Phase 1/2/3 caching and parallelism
