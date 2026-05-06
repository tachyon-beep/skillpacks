---
name: cross-module-flow-analysis
description: Use when the analyzer must propagate lattice values across module / package / project / language boundaries — your application code calls third-party libraries, framework callbacks reach back into your code, plugins load at runtime, or an ML pipeline crosses Python/Rust/C++ boundaries. Covers the module boundary as a first-class lattice operation, function-summary representation at the boundary, hand-written library stubs, framework-callback handling, the precision/soundness/cost triangle for stub libraries, and the boundary-statement discipline. Produces `08-cross-module-flow.md`.
---

# Cross-Module Flow Analysis

## Why Boundaries Need a Sheet of Their Own

`three-phase-inference.md` solves intra- and inter-procedural flow within a body of source the engine can read. The moment the analyzer crosses into code it cannot read — a third-party library, a C extension, an external service, a different language — Phase 3 stops working. Without a discipline at the boundary, the engine has three failure modes:

- **Pessimise everything** — assume every external call is `top`. Sound but useless: most flow ends at the first library call.
- **Optimise everything** — assume every external call is identity. Precise but unsound: vulnerable code shows clean.
- **Inconsistent ad-hoc** — some libraries hand-modelled, most ignored, no record of which is which. Worst of all worlds.

`08-cross-module-flow.md` defines the boundary contract: where the engine's reading stops, what crosses the boundary, what the engine assumes about the other side, and how those assumptions are written down so a developer reading them can decide whether they hold for *their* third-party library.

## What a Module Boundary Is (in the Lattice)

A boundary is any callsite whose callee body is not part of the analyzer's source set. Boundaries include:

| Boundary type | Examples | Body availability |
|---------------|----------|-------------------|
| **First-party** | Same project, different module | Available; flow propagates normally |
| **First-party (planned exclusion)** | A subsystem the team chose not to analyse | Unavailable by policy |
| **Third-party stdlib** | `re`, `os`, `json`, `subprocess` | Available but not analysed; modelled by stub |
| **Third-party non-stdlib** | `requests`, `django`, `numpy`, `sqlalchemy` | Available but typically not analysed; modelled by stub |
| **Native / C extension** | `numpy._core`, `cryptography.hazmat`, anything `.so`/`.pyd` | Body unavailable to the analyzer at all |
| **Foreign-language** | Python calling Rust, Rust calling C, JS calling WASM | Different toolchain altogether |
| **Service** | RPC, REST, queue messages | Not source at all; runtime data |

The lattice value at the boundary's *exit* (the value the caller sees as the call's result and the values written to globals/heaps the caller can observe) is determined by:

1. The lattice values at the *entry* (arguments, captured globals).
2. A **transfer model** for the called function — explicit (a stub) or implicit (a default).
3. The lattice's join semantics if multiple stubs apply (e.g., a method with multiple overloads).

`08-` writes this transfer model down. It does not solve the boundary; it prescribes the discipline by which it is approximated.

## Stubs as the Unit of Boundary Modelling

A **stub** is a hand-written function summary: a tiny program that, given the lattice values of arguments, produces lattice values for return + side effects. Stubs are first-class artifacts in the analyzer; they are versioned, tested, owned, and reviewed.

```python
# Conceptual stub for `subprocess.run(args, ...)`
@stub("subprocess.run")
def stub_subprocess_run(args: LatticeValue, **kwargs) -> StubResult:
    # Sink: any tainted value reaching `args` is a command-injection sink
    if args >= TAINT_UNTRUSTED:
        return StubResult(
            return_tier=TAINT_UNKNOWN,                    # CompletedProcess opaque
            side_effects=[SideEffect.PROCESS_LAUNCHED],
            sinks_reached=[Sink("CWE-78", at="args")],
        )
    return StubResult(return_tier=TAINT_CLEAN, side_effects=[SideEffect.PROCESS_LAUNCHED])
```

A stub captures four things:

1. **Argument flow** — which argument tiers determine the return tier; which arguments are sinks.
2. **Return tier** — `f(a, b, ...)` returns what lattice value as a function of the inputs.
3. **Side effects** — globals written, heap mutations, network/disk/process effects.
4. **Sinks** — argument positions where dangerous flow is detected.

A stub is **sound** if it over-approximates the real function's behaviour: every taint flow the function can carry, the stub also carries; every sink the function exposes, the stub names. A stub is **precise** if it doesn't over-approximate — if it correctly captures sanitisation, narrowed types, or pure-function-ness.

Most stubs over-approximate. The economy is: make the most-trafficked stubs precise (top 50 stdlib functions, top 20 of each load-bearing third-party library), let the rest over-approximate.

## The Default for Unmodelled Calls

For any call to an unmodelled library function, the engine has a **default stub**. Three options, in order of soundness:

### Default A — Pessimistic (sound)

Every unmodelled call returns $\top$, writes $\top$ to every reachable global, and is a sink for every dangerous tier.

**Sound; produces an enormous false-positive rate.** Acceptable only when (a) the analyzer's boundary is highly local (few external calls) or (b) the analyzer is run for a security audit with manual review.

### Default B — Identity (unsound; common)

Every unmodelled call returns the join of its arguments' tiers; no side effects; no sinks.

**Unsound; produces low false-positive rate; misses most cross-library flows.** This is what most analyzers actually do — and most don't admit it. State it explicitly in `08-` if used.

### Default C — Bounded over-approximation (recommended)

Every unmodelled call:

- Returns the **join of its arguments' tiers** (so taint propagates).
- **Writes $\top$** to globals declared as side-effect channels (a small whitelist; everything else assumed pure).
- Sinks **only** if a generic CWE-class rule fires (e.g., a string ending in `execute` / `query` / `eval` / `system` reaching tainted input, regardless of library).

**Mildly unsound; produces moderate false-positive rate; catches most flows.** This is the "honest middle" most production analyzers should declare.

`08-` states which default is in effect and the rationale.

## Stub Library Discipline

A stub library is a set of stubs versioned and shipped alongside the analyzer.

### Sourcing stubs

| Source | Coverage | Quality | Cost |
|--------|----------|---------|------|
| **Hand-written by analyzer team** | Highest-priority libraries (stdlib, top 20 third-party) | High | Linear effort per function |
| **Crowdsourced** (community contributions, e.g., typeshed analogue) | Long tail | Variable | Review burden on the maintainer |
| **Generated from type stubs** (mypy / pyright `.pyi` files) | Type-shape only; no flow semantics | Low for taint, useful for type-flow | Cheap to extract |
| **Generated from doc strings** (LLM-assisted, see `llm-assisted-rule-explanation.md`) | Any documented function | Variable; must be reviewed | Cheap to draft, expensive to verify |
| **Inferred from binary signatures** (C extensions, FFI) | Symbols only; semantics unknown | Lowest; treats as Default C | Free but uninformative |

The mix is a project decision; the discipline is **provenance**: every stub records its source, its author, its review date, and the analyzer version it shipped in.

### Stub versioning

Library APIs change. A stub for `requests==2.27` may not match `requests==2.32`. Two options:

- **Stub-per-version** — pin each stub to a library version range; the engine selects the stub matching the version found in `pyproject.toml` / `requirements.txt`.
- **Stub-per-major** — pin to major versions; document the minor-version drift policy.

Without a versioning model, stubs accumulate as truthful-when-written, lying-by-the-time-anyone-reads-them. State the policy in `08-`.

### Stub testing

Every stub must have a test fixture: a piece of code that calls the modelled function, runs the analyzer, and verifies the lattice value at the call site is what the stub claims. Without a test, the stub is folklore. The fixtures form part of the consistency gate's test corpus (check 11).

## Framework Callbacks and Inversion of Control

Frameworks (Django, Flask, FastAPI, pytest, asyncio) invert control: the framework calls the application's code at points the application code did not declare. The callgraph's edges go *from the framework into the application* at locations the analyzer cannot trivially see.

Three handling strategies:

### Strategy 1 — Synthetic entry points

Recognise framework idioms and synthesise callgraph entries:

- A class extending `django.http.View` has its `get`, `post`, `put`, `delete` methods as entry points with HTTP request parameters.
- A function decorated `@app.route(...)` (Flask) is an entry point with the request object as input.
- A function decorated `@pytest.fixture` is an entry point called with whatever fixtures it requests.

The engine maintains a **framework recognition registry**: framework name → entry-point patterns → input shape. The registry is part of the stub library.

### Strategy 2 — Manifest-declared entry points

Application declares entry points in a manifest (see `manifest-driven-configuration-with-coherence-validation.md`). Useful when framework recognition is incomplete or for in-house frameworks.

### Strategy 3 — Scan-and-discover

The engine walks the AST looking for decorators that match a "looks like a framework binding" heuristic and creates speculative entries. Cheap; noisy; useful for discovery before adding to the registry.

For tier L/XL analyzers, all three strategies coexist: the registry handles known frameworks, the manifest handles in-house ones, and the scan-and-discover output is a debugging aid for adding new framework support.

## Cross-Language Boundaries

Python ↔ Rust (PyO3), Python ↔ C (cffi/ctypes), JS ↔ WASM, JVM ↔ JNI: the FFI boundary is a hard wall.

For an analyzer rooted in one language:

- **The boundary is a stub** — same as any external library, but the body is in another language entirely.
- **The boundary is opaque** — return tier and side effects come from the stub or default.
- **Cross-language type information** — the FFI's declared types (PyO3's `#[pyfunction]` signatures, cffi's headers) inform the stub's argument and return shapes.

For multi-language analyzers (rare but increasing in ML pipelines), the cross-language flow is itself an inference: each language's analyzer produces summaries; a meta-analyzer reconciles them at the FFI boundary. The reconciliation is a join in the *common* lattice — the lattice each language's analyzer agrees to use at the boundary.

This pack does not try to design a multi-language analyzer; it advises that if you need one, the boundary lattice is the architectural foundation and lives in `02-`. See `axiom-pyo3-interop` for the Python ↔ Rust boundary discipline (typed conversion at the boundary, GIL semantics, error mapping); the analyzer-engineering counterpart is to model that boundary as a stub layer.

## The Boundary-Statement Discipline

For each module boundary, `08-` records a one-sentence statement:

```
BOUNDARY: subprocess
  modelled by:        stdlib-stubs/subprocess.py (provenance: hand-written, reviewed alice@team 2026-04, library version: stdlib 3.11+)
  flow semantics:     args -> command-injection sink (CWE-78); env mutations -> top
  precision:          high — full coverage of run, Popen, check_output, getoutput
  open issues:        shell=True interaction with command list; documented in stub

BOUNDARY: requests (third-party HTTP)
  modelled by:        third-party-stubs/requests/ (provenance: hand-written, reviewed bob@team 2026-04, library versions: 2.27.x – 2.32.x)
  flow semantics:     URL/data args -> SSRF sink (CWE-918) when scheme is http*; response.text -> tainted
  precision:          medium — get/post/put/delete covered; session retry semantics not modelled
  open issues:        Auth flow not modelled (treated as identity)

BOUNDARY: numpy
  modelled by:        DEFAULT-C (no stub library)
  flow semantics:     join of arg tiers; pure (no side effects); no sinks declared
  precision:          unknown — numpy operations are mostly pure, but C-extension paths could be exfiltration channels (unlikely but unmodelled)
  open issues:        any operation that goes from numpy back to Python objects (.tolist(), .item()) returns top
```

A developer reading `08-` should be able to answer for any external call:

1. Is this boundary modelled or defaulted?
2. If modelled, by what stub, when reviewed, against which library version?
3. What flow semantics does the model claim?
4. Where are the gaps?

If the answer to (1) is "I don't know," the boundary is undisciplined and the analyzer's findings cross it on a wing and a prayer.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No stub library; every external call defaults | Either massive FPs (Default A) or massive FNs (Default B) | Build a stub library starting with stdlib + top 20 third-party |
| Stubs untested | Stubs drift from library reality silently | Mandatory test fixture per stub; CI runs them |
| Stubs unversioned | Library upgrade breaks analyzer in obscure ways | Pin stub to library version range; CI alerts on drift |
| Default unsound but undeclared | "We catch everything" rhetoric, factual under-approximation | State the default in `08-`; gate consistency check 5 |
| Framework callbacks not modelled | Findings stop at the application boundary; entry points invisible | Synthetic entry points via framework registry + manifest |
| C extensions treated as opaque without policy | Native libraries' sinks invisible | Either stub the C extension's surface (FFI signatures) or declare the boundary as Default C |
| Cross-language flow handwaved | Multi-language pipelines: untracked exfiltration paths | Boundary lattice in `02-`; FFI as a stub layer; per-language summaries |
| Stub provenance unrecorded | Cannot tell which stubs are reviewed, which are LLM-drafted, which inherited | Provenance fields on every stub; review-date metadata |
| Stub set frozen | New high-traffic library not modelled for years | Stub-coverage metric; CI flags top-N unmodelled call counts |

## Stub-Coverage Metric

A first-class CI metric:

```
stub_coverage = stubbed_callsites / total_external_callsites
target_rate >= 0.70

stub_top_unmodelled = "the 10 most-called external functions without stubs"
   reported per release; informs roadmap
```

Action when below target: prioritise stubs for the top unmodelled call counts. The budget is the same shape as the suppression budget in `false-positive-economics.md`.

## The Decision Output (`08-cross-module-flow.md`)

A complete `08-` answers:

1. **Boundary inventory** — list of boundary types in scope; which are stubbed, which defaulted.
2. **Default policy** — A / B / C; rationale; explicit declaration of unsoundness if B.
3. **Stub library structure** — directory layout, ownership, review process.
4. **Stub schema** — argument flow shape, return tier function, side effects, sinks.
5. **Stub provenance fields** — author, review date, library version range, source (hand-written, generated, LLM-drafted).
6. **Stub testing contract** — every stub has a fixture; CI runs them; fail-on-drift policy.
7. **Framework recognition registry** — which frameworks are recognised; in what idioms; with what synthetic entry points.
8. **Manifest-declared entry points** — schema; cross-link to `manifest-driven-configuration-with-coherence-validation.md`.
9. **Cross-language policy** — if any FFI is in scope, the boundary lattice and the per-language summary contract.
10. **Boundary-statement format** — the one-sentence-per-boundary discipline; example.
11. **Stub-coverage metric and budget** — target rate, top-unmodelled report cadence, action on shortfall.

## Cross-References

- `taint-lattice-design.md` — the lattice values stubs produce; the boundary lattice if multi-language
- `three-phase-inference.md` — Phase 2 summaries are the in-engine analogue of stubs; same discipline
- `callgraph-construction.md` — boundaries are unresolved-callee sites; default callgraph behaviour
- `plugin-architecture-for-analyzer-rules.md` — generic CWE-class rules that fire on Default C even without a stub
- `false-positive-economics.md` — many FPs are over-pessimised stubs; many FNs are under-pessimised stubs; the budget includes stub quality
- `manifest-driven-configuration-with-coherence-validation.md` — manifest entries for in-house framework entry points and stub overrides
- `decorator-as-assertion.md` — when a framework's decorator is also a runtime assertion, the stub records both sides
- `llm-assisted-rule-explanation.md` — the same pattern applies to LLM-drafted stubs (drafted, then human-reviewed, never trusted unreviewed)
- Cross-pack: `axiom-pyo3-interop:typed-conversion-at-the-boundary` — the Python ↔ Rust FFI conversion discipline that informs cross-language stub design
- Cross-pack: `axiom-system-archaeologist:analyze-dependencies` — consumes the boundary inventory to draw module dependency graphs
