---
name: callgraph-construction
description: Use when Phase 3 of the inference pipeline (`three-phase-inference.md`) needs an actual callgraph and the language has features that make construction non-trivial — virtual dispatch, dynamic imports, decorators that rewrite the callable, monkey-patching, `getattr`/`__getattr__`, `eval`, callable objects, plug-in loaders. Covers resolution strategies (name-based, type-based, points-to-driven), the conservativeness ladder, monomorphisation, dynamic-feature handling, and how the chosen resolution depth shows up as a soundness/completeness statement in `99-`. Produces `07-callgraph-construction.md`.
---

# Callgraph Construction

## Why a Sheet for Just the Callgraph

`three-phase-inference.md` assumes a callgraph exists. In every language anyone writes static analyzers for, building that callgraph is the part that decides whether the analyzer is sound, fast, or even useful. A callgraph is not a derivable artifact — it is a *design choice* dressed up as a graph, and the choice you make in `07-` propagates into every flow finding the analyzer ever emits.

Most analyzer projects spend their first six months pretending callgraph construction is "just" a name-resolution problem, then spend the next two years discovering that virtual dispatch, dynamic imports, decorators, callables-as-data, and `getattr` are first-class language features whose semantics the engine cannot ignore without lying.

`07-callgraph-construction.md` is where you pick the resolution strategy, state its conservativeness, and write down which language features you handle precisely, which conservatively, and which the engine declares out of scope.

## What the Callgraph Must Answer

For each callsite in the program, the callgraph answers two questions:

1. **Which functions could this callsite reach?** (forward edge)
2. **Which callsites could reach this function?** (reverse edge — needed for incremental invalidation; see `scaling-to-large-codebases.md`)

A callgraph is a relation `CallSite → Set[Function]` with a soundness claim. The claim is one of:

| Claim | Meaning | Cost |
|-------|---------|------|
| **Sound (over-approximation)** | If a callsite *could* reach `f` at runtime, the graph contains the edge. May contain extra edges. | Cheaper to compute; produces FPs in dataflow rules |
| **Precise (under-approximation)** | Every edge is reachable at runtime. May miss edges. | More expensive; produces FNs in dataflow rules |
| **Sound *and* precise** | Every reachable edge present, no spurious edges. | Undecidable in general (Rice); achievable only for simple subsets |
| **Best-effort** | "We try, we sometimes miss, we sometimes over-include." | Cheapest; useless for security-bearing analyzers |

State the claim explicitly in `07-`. "Best-effort" is a confession, not an architecture.

## The Resolution Ladder

Resolution strategies form a ladder; pick the rung that matches your tier (`Analyzer Tier` in the router). Each rung is more precise and more expensive than the last.

### Rung 0 — Name resolution only (CHA-equivalent for OO)

Treat every call `f(...)` as edge to all functions named `f` visible in scope. For `obj.method(...)`, edge to *every* method named `method` on *any* class in the program.

**Strengths:** trivial; fast; sound for most non-OO Python.
**Weaknesses:** explosive over-approximation in OO codebases (every `__str__` call reaches every `__str__` defined anywhere); false positives by the thousand for taint analysis.

Acceptable for tier S analyzers without dataflow rules.

### Rung 1 — Class Hierarchy Analysis (CHA)

For `obj.method(...)` where `obj` has a declared or inferred class `C`, edge to `method` on `C` and on every subclass of `C` that overrides it.

**Strengths:** order-of-magnitude smaller than Rung 0 for typed codebases.
**Weaknesses:** still over-approximates anywhere typing information is missing or imprecise. In Python, "class of `obj`" is a runtime question; CHA needs at least nominal typing (annotations or inference) to be useful.

Default rung for tier M analyzers in typed languages; in Python, often paired with mypy-style inference for the class context.

### Rung 2 — Rapid Type Analysis (RTA)

CHA, but restrict to classes actually instantiated somewhere in the program. A method `C.method` is a target only if `C` (or a subclass) is constructed somewhere reachable.

**Strengths:** cuts CHA's edge set roughly in half for libraries with abstract base classes that are never directly used.
**Weaknesses:** "instantiated" requires reachability, which requires a callgraph — the chicken-and-egg is resolved by an iterated worklist.

Default rung for tier L analyzers.

### Rung 3 — Variable Type Analysis (VTA) / Points-to

For each receiver expression, compute the set of objects (allocation sites or symbolic abstract objects) it can hold; resolve calls to the methods of those objects' classes.

**Strengths:** considerably more precise than RTA; handles polymorphism through typed containers, fields, returns.
**Weaknesses:** points-to is a non-trivial dataflow problem in its own right (Andersen-style is `O(n³)`, Steensgaard is `O(n·α(n))` but coarser). Implementing it badly is worse than not implementing it.

Tier L/XL analyzers with security obligations. Below tier L, the cost rarely pays back.

### Rung 4 — Whole-program control-flow analysis (k-CFA, m-CFA)

Context-sensitive points-to; a call is resolved with knowledge of the call stack (or a bounded slice of it). `0-CFA` is context-insensitive (Rung 3); `1-CFA` keeps one frame of context; `k-CFA` keeps `k`.

**Strengths:** precision approaches "exactly the runtime callgraph" for non-pathological programs.
**Weaknesses:** exponential in `k`. `1-CFA` is sometimes affordable; `2-CFA` rarely is on real codebases without aggressive abstraction (object-sensitivity, type-sensitivity).

Research-grade or specialised security tools; rarely the right rung for a production analyzer. Document if used.

**Picking your rung:**

| Tier (router) | Default rung | Notes |
|---------------|-------------|-------|
| XS / S | 0 or 1 | Often no callgraph needed at all |
| M | 1 (CHA) | Good fit if typing is available |
| L | 2 (RTA) | The first rung where dataflow precision pays back |
| XL (security) | 3 (VTA) | Or 4-CFA if formal precision claims required |

## The Conservativeness Floor

Whatever rung you pick, you will encounter language features the rung doesn't handle. The conservative move is to **add an over-approximating edge to a synthetic "any function" node** (often called `top` or `unknown_callee`) for every callsite the resolver could not classify. This:

- Keeps the graph **sound** (every real edge is present, even if represented coarsely).
- Lets dataflow rules cleanly express "this callsite reaches `top`, treat as $\top$ in the lattice".
- Surfaces unresolved callsites as a first-class metric (the **resolution rate**: `resolved_callsites / total_callsites`).

A resolution rate below 80% is the symptom of an analyzer that is mostly guessing. State the floor in `07-`; gate on it in CI.

## Dynamic Features and What to Do About Them

The features below are why callgraph construction is hard in dynamic languages.

### Virtual dispatch / method override

Resolved by Rung 1+. Soundness floor: edge to every override at every callsite where the receiver type is unknown.

### Dynamic imports (`importlib`, `__import__`, conditional `import`)

`__import__('mod_' + var)` is unresolvable without runtime values. Three responses:

- **Whitelist** — declare which dynamic-import patterns are allowed and resolved (e.g., the engine recognises `importlib.import_module(LITERAL)` and resolves it). Anything else → `top`.
- **Plug-in registry** — recognise specific framework idioms (Django app loading, pytest plugin discovery, `entry_points`) and treat them as registered call edges from synthetic loader sites.
- **Surrender** — emit a finding at the dynamic-import site: "callgraph cannot resolve dynamic import; flow downstream of this point is unmodelled". This is not a bug; it is honesty.

### `getattr(obj, name)` / `__getattr__` / `__call__`

`getattr(obj, dynamic_name)` is a value-level lookup. Without knowing `dynamic_name`, the resolver must edge to every attribute named anything on `obj`'s class hierarchy, which is Rung 0 behaviour for that callsite.

If `dynamic_name` is a literal (`getattr(obj, "method")`), treat it as `obj.method`. This case is common; recognise it.

`__getattr__` (the catch-all) is a method whose body decides what gets returned. If the body returns a callable, every "not-found attribute access" is potentially a call. The conservative move: every attribute access on a class with `__getattr__` is `top` unless the engine can statically narrow the body.

### Decorators that rewrite the callable

A decorator may return a wrapper, a different function entirely, or a class instance. The "callable" the user invokes is not the function whose body the decorator decorated.

Two stances:

- **Decorator-aware resolution** — the engine has a model for known decorators (`@functools.wraps`, `@staticmethod`, `@classmethod`, `@property`, common framework decorators) and resolves through them.
- **Decorator-opaque** — every call to a decorated function is a call to "the decorator's wrapper"; the engine sees only that wrapper, not the underlying function.

The aware stance requires a registry (decorator name → resolution shape); the opaque stance under-approximates flow through decorators. `decorator-as-assertion.md` covers the *design* of decorators that play well with both static and runtime; this sheet handles the *resolution* of decorators in the wild.

### Monkey-patching (`MyClass.method = new_method` after class definition)

In a soundness-bearing analyzer, monkey-patching is the worst-case enemy: it invalidates every callgraph edge through `MyClass.method`. Standard responses:

- **Forbid by rule** — emit a high-severity finding at every `Class.method = ...` assignment. This is a static rule on top of the analyzer.
- **Pessimise from the patch site** — once a class is detected as monkey-patched, every method on that class is `top` for the analysis.
- **Require a manifest entry** — monkey-patches must be registered in a manifest with their replacement targets; the engine reads the manifest. (See `manifest-driven-configuration-with-coherence-validation.md`.)

### `eval`, `exec`, `compile`

Unresolvable in general. The engine should treat these as `top` callsites and additionally fire a high-severity rule at every reachable `eval`/`exec` (the security implications are usually independent of the callgraph).

### Higher-order functions (`map(f, xs)`, `partial`, etc.)

The callee is a value, not a name. Resolve as in Rung 3 (points-to on the callable variable). The result is a set of callees; the engine emits an edge from the higher-order callsite to each. `cross-module-flow-analysis.md` covers higher-order flow at module boundaries.

### Callable objects (`__call__`)

`obj(...)` where `obj` has a `__call__` method. Treat as `obj.__call__(...)`. Rung 1+ handles this once `obj`'s class is known.

## Monomorphisation

For polymorphic methods (generic in the type system, or duck-typed in dynamic languages), the engine has two choices:

- **Monomorphic callgraph** — one edge per `(callsite, callee)` pair; receiver type is collapsed.
- **Monomorphised callgraph** — one edge per `(callsite, callee, receiver_type)` triple; the same callsite produces multiple edges if the receiver type varies.

Monomorphisation costs more (graph size up to `O(types × callsites)`) but lets dataflow rules differentiate findings by receiver type. This matters when the same method has different security semantics on different types — `cursor.execute` on a `RawCursor` is different from `cursor.execute` on a `SafeWrappedCursor`.

For tier L/XL analyzers, monomorphise; for tier M and below, don't bother.

## Construction Algorithm (Iterated Refinement)

Callgraph construction interacts with Phase 3: the callgraph informs propagation, but propagation can refine the callgraph (a flow shows that this `getattr` call only reaches `method_a`, not `method_b`).

The standard structure:

```python
def construct_callgraph(program: Program, rung: ResolutionRung) -> CallGraph:
    # Initial graph: every callsite gets edges per the static rung
    graph = initial_resolution(program, rung)
    # Iterate: dataflow may narrow receiver types; receiver types may narrow edges
    changed = True
    while changed:
        types = run_phase_1_and_2(program, graph)        # uses current graph
        new_graph = refine_resolution(program, rung, types)
        changed = (new_graph != graph)
        graph = new_graph
    return graph
```

**Termination:** the graph monotonically loses edges (refinement only narrows); edge set is finite; therefore the loop terminates in at most `|edges|` iterations. In practice, two or three rounds suffice for most programs.

**Soundness:** at every iteration, the graph is a valid over-approximation; refining it removes only edges that dataflow proved infeasible. The fixed-point graph is the most precise refinement of the initial rung.

`07-` should specify whether iterated refinement is enabled (it costs a 2–3× factor over single-pass) and what the fixed-point criterion is.

## The Resolution-Rate Metric

A first-class CI metric:

```
resolution_rate = resolved_callsites / total_callsites
target_rate     >= 0.85   # configurable per project
```

Action when below target:

1. **Investigate** — bucket unresolved callsites by reason (dynamic import, `getattr` on dynamic name, monkey-patch, `eval`).
2. **Refine** — handle a recognised pattern (e.g., add `getattr(obj, LITERAL)` resolution).
3. **Pessimise** — accept the unresolved fraction; the dataflow rules using `top` callees will produce conservative findings at those sites, which is correct.
4. **Forbid** — emit a rule at the unresolved pattern (e.g., disallow non-literal dynamic imports outside the plugin loader).

The resolution rate is the analyzer's honesty meter. A "100%-resolved" callgraph either reflects a trivial codebase or hides Rung 0 over-approximation.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Treating "no edge found" as no-call | Massive false negatives at decorators, dynamic dispatch | Conservative floor: unresolved callsite → edge to `top` |
| Resolving `getattr(obj, LITERAL)` like `getattr(obj, var)` | False negatives on a common pattern | Special-case literal-string `getattr`; same for `setattr`, `hasattr` |
| Monkey-patching ignored | Findings stale after a `Class.method = ...` | Forbid by rule, or pessimise from the patch site |
| Decorators always opaque | Flow through decorators invisible | Decorator-aware resolution registry; common framework decorators recognised |
| Single-pass resolution after dataflow | Engine never refines the callgraph from inferred types | Iterated refinement loop with monotonic-shrinkage termination |
| No resolution-rate metric | Engine is honest internally but the team doesn't see degradation | Resolution rate as a CI metric; budget threshold |
| Higher-order callsites resolved by name | Spurious edges or missed edges | Resolve via points-to on the callable variable; produce edges to each callable's class |
| `eval`/`exec` callgraph-modelled | Soundness lie | Treat as `top` and emit a rule; do not pretend to resolve them |

## Soundness/Completeness Statement

`07-` ends with an explicit statement consumed by the consistency gate (check 5):

```
RESOLUTION CLAIM (07-)

Strategy: <Rung 0|1|2|3|4>
Soundness: every reachable edge present in the graph at fixed point,
   with the following declared exceptions:
     - eval / exec / compile callsites: top
     - getattr/setattr with non-literal name: top
     - monkey-patched classes: top from the patch site forward
     - decorators not in the recognition registry: opaque (edge to wrapper)
Precision: declared exceptions noted; resolution rate target ≥ 0.85;
   refinement loop iterated to monotonic fixed point.
Action on unresolved: top callee + finding at the unresolvable site
   (rule STA-RES-001 or equivalent).
```

If `99-` claims soundness for the analyzer, the `07-` claim must support it. Check the chain: the lattice is sound (`02-`); the inference is sound given the lattice (`03-`); the callgraph is sound given the resolution claim (`07-`). Break any link, the chain fails.

## The Decision Output (`07-callgraph-construction.md`)

A complete `07-` answers:

1. **Resolution rung** — which of 0–4; rationale tied to analyzer tier.
2. **Soundness vs precision claim** — explicit, with the declared exception list.
3. **Dynamic-feature policy** — for each of: virtual dispatch, dynamic imports, `getattr`/`__getattr__`, decorators, monkey-patching, `eval`/`exec`, higher-order, callable objects.
4. **Monomorphisation** — yes / no; rationale.
5. **Iterated refinement** — yes / no; termination criterion.
6. **Synthetic `top` node** — defined, distinguished from real functions, lattice value at calls to it.
7. **Resolution rate** — target threshold; CI gate behaviour on breach.
8. **Manifest entries** — for monkey-patching declarations and decorator registry; cross-link to `manifest-driven-configuration-with-coherence-validation.md`.
9. **Output schema** — how the callgraph is serialised (for consumption by `system-archaeologist`, debugging, replay).
10. **Test corpus** — fixtures exercising every dynamic-feature category; consistency gate check 11.

## Cross-References

- `ast-visitation-patterns.md` — supplies the AST that callsites are collected from
- `taint-lattice-design.md` — the `top` lattice value used at unresolved callsites
- `three-phase-inference.md` — Phase 3 consumes the callgraph this sheet builds
- `cross-module-flow-analysis.md` — extends callgraph construction across module boundaries with library stubs
- `decorator-as-assertion.md` — design discipline for decorators that play well with the resolution registry
- `manifest-driven-configuration-with-coherence-validation.md` — declarations for monkey-patches, decorator registry, plugin loaders
- `scaling-to-large-codebases.md` — incremental callgraph invalidation; reverse-edge index
- `false-positive-economics.md` — many "FPs" trace back to over-approximate callgraph edges; check `07-` before refining the rule
- Cross-pack: `axiom-system-archaeologist:analyze-dependencies` — consumes callgraph output for system-level dependency analysis
