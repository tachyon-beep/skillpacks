---
name: decorator-as-assertion
description: Use when designing decorators (Python), attributes (C#), annotations (Java), proc-macros (Rust) that are *both* a runtime check and a static rule trigger — `@authenticated`, `@validates(schema)`, `@requires_capability(X)`, `@idempotent`, `@pure`. Covers the agreement contract between runtime and static enforcement, the descriptor-pattern implementation that keeps the decorated callable introspectable by the analyzer, the metadata schema that makes the decorator a first-class lattice/rule input, and the failure modes when runtime and static disagree about what was enforced. Produces `09-decorator-as-assertion-spec.md`.
---

# Decorator as Assertion

## What This Sheet Is About

A common pattern in production systems: a decorator decorates a function, and *both* the runtime and the static analyzer treat it as a load-bearing assertion.

```python
@requires_capability("admin")
def delete_user(user_id: int) -> None:
    ...
```

At runtime, `@requires_capability` raises if the caller lacks the capability. The decorator is doing real work: it is the gate.

At static-analysis time, the analyzer sees the decorator and treats `delete_user` as a function whose flow has been narrowed: callers without the static capability annotation are flagged, and the body of `delete_user` can assume `caller_capabilities >= {admin}` for the rest of its analysis.

When this pattern is designed coherently — runtime check and static rule grounded in the same metadata, agreeing on what the decorator means — it is one of the most powerful tools in the analyzer's repertoire. When it isn't designed coherently, it is a high-confidence source of bugs that look like neither: runtime says one thing, static says another, and the gap is where exploits live.

`09-decorator-as-assertion-spec.md` is where you commit to the agreement contract between runtime and static and to the implementation discipline that keeps them aligned.

## The Agreement Contract

A decorator-as-assertion has *one* meaning in two enforcement surfaces. The agreement contract is the formal statement of that meaning.

```
DECORATOR: @requires_capability(cap)
   meaning:
      "Calls to the decorated function are valid only when the caller's
       capability set is a superset of {cap}."
   runtime enforcement:
      raise PermissionError if cap not in current_capabilities() at call time
   static enforcement:
      callsites whose lattice value at `current_capabilities` does not
      include `cap` are flagged (rule STA-CAP-001)
   in-body assumption:
      within the body of the decorated function, `current_capabilities`
      includes `cap` (lattice narrowed)
   metadata stored on the callable:
      __required_caps__ : frozenset[str]
   stub model (cross-link 08-):
      the decorator wrapper is transparent; resolution unwraps to the
      underlying function's body for callgraph and dataflow
```

Every decorator-as-assertion gets such a block in `09-`. The block is the contract; the runtime and static sides are derived implementations of it. Drift between the two is a contract violation, not an implementation detail.

## Why This Pattern Is Hard

Three failure modes recur:

### Failure 1 — Runtime and static disagree about what the decorator means

The runtime checks `caller has cap == admin`; the static rule checks `callsite is annotated as `admin``. Capabilities ≠ annotations: an unannotated callsite passes the runtime check (because the caller is admin at runtime) but fails the static check, or vice versa. The team papers it over with suppressions.

**Cause:** the contract was never written down. Each side was implemented against its own intuition. Drift is invisible until production.
**Fix:** the contract block. Both sides cite it.

### Failure 2 — The decorator hides the function from the analyzer

A naïve `@requires_capability` returns a wrapper that calls the underlying function. The wrapper is what the engine sees. The analyzer cannot reach the body, sees no flow, and emits no findings inside it.

```python
# WRONG: wrapper hides the body
def requires_capability(cap):
    def deco(fn):
        def wrapper(*a, **kw):
            check(cap)
            return fn(*a, **kw)
        return wrapper          # the body of fn is now invisible to analyzers without unwrap support
    return deco
```

**Fix:** descriptor pattern (below) or `functools.wraps` plus an analyzer-recognised unwrap convention (`__wrapped__`).

### Failure 3 — The decorator's metadata is not on the decorated callable

The static rule needs to know `cap == admin` at the *callsite*. If the decorator stored `cap` only in a closure inside the wrapper, the analyzer cannot recover it without value-level reasoning. The runtime knows; the analyzer is locked out.

**Fix:** decorator stores its parameters as attributes on the decorated callable: `fn.__required_caps__ = frozenset({cap})`. Now any analyzer that reads attribute literals can recover the metadata.

## The Descriptor / Wrapper Discipline

For the analyzer to model the decorator, the decorator must be implementable in a way that:

1. Preserves the underlying function's signature, name, and docstring (`functools.wraps` or `__wrapped__`).
2. Stores its parameters as introspectable attributes on the decorated callable.
3. Does not interpose computation that the analyzer cannot model (no `eval`, no dynamic class creation, no monkey-patching of callers).

```python
# RIGHT: the decorator is transparent and introspectable
def requires_capability(cap: str):
    def deco(fn):
        @functools.wraps(fn)                          # preserves __name__, __qualname__, __doc__, __wrapped__
        def wrapper(*a, **kw):
            if cap not in current_capabilities():
                raise PermissionError(f"requires {cap}")
            return fn(*a, **kw)
        # Stored on the wrapper AND on the original via functools.wraps' __wrapped__ chain
        wrapper.__required_caps__ = frozenset({cap}) | getattr(fn, "__required_caps__", frozenset())
        return wrapper
    return deco
```

For a class-decorator or attribute-decorator that needs descriptor semantics (binding to instances), the same discipline applies; the metadata lives on `__func__` or on the descriptor itself. The analyzer's decorator registry (see `callgraph-construction.md`) names which attribute holds the metadata.

## Stacking Decorators

`@requires_capability("admin") @validates(schema=UserDelete)` — both decorators apply.

The runtime composes them in the natural order; the static rule has to do the same. The discipline:

- Each decorator's metadata is stored under a **distinct, non-colliding attribute** (`__required_caps__`, `__validation_schema__`).
- The metadata for each decorator includes its **position in the stack**, so an analyzer can reconstruct order if it matters (rare, but matters for some checks).
- `functools.wraps` chains `__wrapped__` so the analyzer can find the underlying function at the bottom of the stack.

If two decorators write to the same attribute (e.g., two different "requires" decorators that both write `__requires__`), the metadata silently collides. The discipline: attribute names are namespaced by decorator (`__required_caps__`, not `__caps__`).

## The Engine's Side

For the analyzer to consume decorator-as-assertion, three pieces of machinery are needed.

### Decorator recognition registry

A static configuration: `decorator name → metadata-extraction recipe → static-rule-trigger`. Lives in the manifest (see `manifest-driven-configuration-with-coherence-validation.md`).

```yaml
decorators:
  - name: app.security.requires_capability
    aliases: [requires_capability, requires_cap]
    metadata_attr: __required_caps__
    arg_extraction: positional[0]                # cap is the first positional arg
    rule_trigger: STA-CAP-001
    in_body_assumption: extends current_capabilities lattice value
    transparency: wrap (analyzer follows __wrapped__)
```

### Lattice extension at the body entry

When Phase 1 begins analysing a function body, it consults the decorator registry. For each recognised decorator on the function, the lattice value at the entry point is narrowed per the decorator's `in_body_assumption`.

```python
def initial_env_for(fn: Function) -> Env:
    env = bottom_env()
    for dec in fn.decorators_recognised:
        env = dec.in_body_assumption(env)            # narrows env per decorator semantics
    return env
```

This is the static counterpart to "inside the function body, the runtime check has succeeded; we may assume the precondition."

### Rule firing at the callsite

Phase 3 sees a callsite to a function with `__required_caps__ = {"admin"}`. The static rule consults the lattice at the callsite's `current_capabilities` variable, joins, and fires if the rule's predicate is violated.

```python
@rule(id="STA-CAP-001", severity="error", category="security")
def capability_check(ctx: AnalysisCtx, callsite: CallSite) -> Iterator[Finding]:
    required = getattr(callsite.callee, "__required_caps__", frozenset())
    if not required:
        return
    held = ctx.lattice_value_at(callsite, "current_capabilities")
    missing = required - held.as_set()
    if missing:
        yield Finding(
            location=callsite.location,
            message=f"Call to {callsite.callee.name} requires capabilities {missing}, none held statically",
            rule="STA-CAP-001",
        )
```

The rule is a normal plugin (see `plugin-architecture-for-analyzer-rules.md`); the trick is that its inputs are decorator metadata, which became inputs because the decorator was designed to expose them.

## Disagreement Modes

The runtime check fires; the static rule does not (or vice versa). Each is a contract violation, but with different fault attribution.

| Disagreement | Cause | Resolution |
|--------------|-------|------------|
| Runtime fires, static silent | Static lattice doesn't track this capability; or capability is data-dependent | Refine the lattice (`02-`) to track capability, or move the check to runtime-only |
| Static fires, runtime silent | Static is over-approximating; runtime check is being satisfied by paths static can't see | Refine callgraph (`07-`) or make the runtime path visible (e.g., manifest entry); or accept FP per `false-positive-economics.md` |
| Both silent on a known-violating path | Both are wrong | Test corpus must cover this case; both implementations must be fixed; this is the worst outcome |
| Both fire on a known-clean path | Decorator is wrong; underlying check is too strict | Fix the decorator's contract |

`09-` records the team's policy on each disagreement type. The default policy: runtime is authoritative for runtime safety; static is authoritative for advisory rules (early warning). Treat any disagreement as a contract review trigger.

## Decorator-as-Assertion in Other Languages

The pattern generalises. In each case, `09-` adapts the same contract template.

| Language | Mechanism | Metadata location |
|----------|-----------|-------------------|
| **Python** | Decorator returning wrapped function | Attributes on the callable (`__required_caps__`) |
| **C#** | Attribute on method | `[RequiresCapability("admin")]` — read by reflection / Roslyn analyzer |
| **Java** | Annotation on method | `@RequiresCapability("admin")` — read by reflection / annotation processor |
| **TypeScript** | Decorator function (experimental) | Reflect.metadata or per-decorator convention |
| **Rust** | Proc-macro generating wrapper + metadata | Generated `static` items the analyzer reads |
| **Go** | Convention (no native decorators) — function-pair pattern (`requireCapability(cap, fn)`) | Caller-side argument; no decorator metadata; harder for static |

In every case, the discipline is the same: contract written down once, runtime and static derive from the same metadata, the analyzer's recognition registry knows what to look for.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Decorator stores parameters in a closure only | Static analyzer cannot recover the parameters; rule cannot fire | Store on the callable as a `frozenset` / `dict` attribute |
| `@functools.wraps` omitted | Analyzer loses signature, docstring, `__wrapped__` | Always `@functools.wraps(fn)`; CI rule against wrappers without it |
| Decorator wraps an opaque callable (e.g., partially applied) | `__wrapped__` chain is broken | Decorator preserves `__wrapped__` if it exists; analyzer follows the chain |
| Two decorators write to the same metadata attribute | Silent metadata collision | Namespace metadata attribute names per decorator |
| Runtime check and static rule diverge in semantics | Latent disagreement; security gaps | Single contract block in `09-`; both implementations cite it |
| Analyzer doesn't recognise the decorator | Decorator is invisible; rule never fires | Decorator name in the recognition registry (manifest) |
| Stacked decorators have order-sensitive semantics but no order metadata | Order-dependent disagreement | Position-in-stack stored alongside metadata |
| Decorator does dynamic class creation or monkey-patching | Analyzer cannot model; entire decorated class becomes opaque | Use decorators that don't rewrite the world; descriptor pattern over class swapping |
| Decorator-as-assertion used for invariants statics can't decide | Inevitable disagreement; user blames the analyzer | `06-static-vs-runtime-tradeoffs.md` first; if runtime-only, document and don't fake a static rule |

## Test Corpus Discipline

For every decorator-as-assertion, the test corpus must contain four fixtures:

1. **TP — caller violates the contract; rule must fire.**
2. **TN — caller satisfies the contract via a path the lattice can see; rule must not fire.**
3. **FP-known — caller satisfies the contract via a path the lattice cannot see; rule fires; documented as a known FP requiring suppression with justification.**
4. **FN-known — caller violates the contract through a path the static rule cannot reach (e.g., through `eval`); rule does not fire; documented as a known FN with the rationale "runtime is authoritative here".**

Without these, the rule is unfalsifiable. Cases 3 and 4 are honesty: they record the gap between the static and runtime sides explicitly, so future maintainers don't think the rule is a complete enforcement.

## The Decision Output (`09-decorator-as-assertion-spec.md`)

A complete `09-` answers:

1. **Decorator inventory** — every decorator-as-assertion in scope; its meaning.
2. **Contract block per decorator** — meaning, runtime enforcement, static enforcement, in-body assumption, metadata location, stub model.
3. **Implementation discipline** — `functools.wraps`, attribute namespacing, descriptor pattern when applicable.
4. **Recognition registry entry** — manifest fragment for each decorator.
5. **Lattice-extension semantics** — how the in-body assumption modifies Phase 1 entry env.
6. **Rule mapping** — which static rule fires for each decorator's violation.
7. **Disagreement policy** — what happens when runtime and static diverge; authority assignment.
8. **Stacking semantics** — how multiple decorators compose; metadata namespacing.
9. **Cross-language note** — if any non-Python decorators exist, the analogous mechanism.
10. **Test corpus** — TP, TN, FP-known, FN-known per decorator; cross-link to `false-positive-economics.md` for the FP/FN registry.

## Cross-References

- `taint-lattice-design.md` — the lattice values the decorator narrows or sinks against
- `three-phase-inference.md` — Phase 1 entry-env modification per recognised decorators
- `plugin-architecture-for-analyzer-rules.md` — the static rule that fires on contract violation
- `false-positive-economics.md` — known FPs and FNs are part of the budget
- `static-vs-runtime-tradeoffs.md` — the prior question: should this be static at all?
- `callgraph-construction.md` — decorator-aware resolution; the recognition registry
- `cross-module-flow-analysis.md` — when a decorator from a third-party library is the assertion; framework decorators
- `manifest-driven-configuration-with-coherence-validation.md` — the recognition registry's home
- Cross-pack: `axiom-audit-pipelines:decision-provenance` — when the decorator is a regulator-visible control, the contract is itself an auditable decision
- Cross-pack: `ordis-security-architect:design-controls` — capability decorators map to control families; align the rule taxonomy
