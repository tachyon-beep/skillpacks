---
name: dependency-direction
description: Use when deciding what belongs in a shared contracts package, when a contract module needs "just one" import from a subsystem, when contracts and a subsystem have developed an import cycle or must deploy together, when a shared enum or helper is being duplicated to avoid an import, or when a generated/IDL contract package pulls in subsystem code. Covers the leaf-package rule, what contracts may and may not contain, a working import-linter CI gate, the leak paths that survive the gate, and the codegen equivalent.
---

# Dependency Direction

**The contract layer imports nothing from any subsystem: every subsystem imports it, it imports none of them. Direction held by convention is direction already lost — it holds only when an import-lint gate fails the build.**

## When this earns its cost

Read this sheet when:

- You are creating a shared `contracts/` package in a multi-subsystem repository and deciding what goes in it.
- A contract module wants to import a subsystem — for an enum, a validation helper, a type alias, a default-picker.
- Contracts and one subsystem have become impossible to release independently, or the import graph has a cycle through the contract package.
- A "shared utilities" module has appeared next to (or below) contracts and nobody can say which way it points.
- Contract classes are generated from proto/IDL and the generated package imports handwritten application code.
- A review is about to wave through an import because "it's only for type checking".

## Why the direction is the whole architecture

Contracts are the shared language subsystems use to speak to each other. A language works because it is *below* every speaker: `ingest`, `scheduler`, and `api` can each be rewritten, replaced, or deleted without the other two noticing, precisely because the only thing they hold in common is a package that holds nothing in common with them.

One import from the contract layer into `ingest` destroys that in three ways at once:

- **Transitive coupling.** `scheduler` imports contracts; contracts imports `ingest`; `scheduler` now loads `ingest` and everything `ingest` pulls in — its client libraries, its config module, its import-time side effects. A subsystem that no team believed was involved is now load-bearing for every other subsystem in the repository.
- **Cycles.** `ingest` imports contracts (it must — it speaks the language) and contracts now imports `ingest`. Python tolerates the cycle right up until an import order changes and a partially-initialised module raises `AttributeError` in production, at the boundary, on a code path that worked yesterday.
- **Deploy coupling.** Two packages in a cycle version together and ship together. The independent-release property that made contracts worth extracting is gone, and the discovery usually happens mid-release.

The fourth consequence is the expensive one, and it is not about imports at all. **An import is a channel for policy to leak into schema.** The moment contracts can call `ingest.defaults.pick_default_region(...)`, the contract layer stops *defining what may be said* and starts *deciding an individual case* — and it decides it using one subsystem's preferences, on behalf of every subsystem, invisibly, with no policy version attached. Contract review no longer sees the decision; nobody reviews `ingest`'s default-picker as if it were schema. This is `deterministic-resolution.md`'s mechanism-versus-policy test applied to the package graph: the direction rule is what keeps the answer mechanically checkable.

## What the contract layer may contain

**May contain** — all of it pure, all of it about *form*:

- Record type definitions: frozen dataclasses, `TypedDict`s, Pydantic models, generated message classes.
- Closed enums for every constrained field, including absence reasons and outcome codes.
- Validation at construction: field presence, ranges, cross-field invariants, unit and version checks — the parse-time refusals in `silent-default-elimination.md`.
- Canonical serialization and canonical hashing (`canonical-identity.md`).
- Pure deterministic resolution *mechanism*: intersection, narrowing, canonical tie-breaking, alias→canonical mapping.
- Version constants, the version gate's supported-version set, and the exception types (`ContractViolation`) that boundaries raise.

**May never contain:**

- Subsystem policy: thresholds, preferences, weights, defaults, which-candidate-wins logic. These live in versioned policy records that resolvers take as *input* (`versioned-policy-parameters.md`).
- Case-specific decisions: anything that looks at one record and picks an outcome on a subsystem's behalf.
- I/O of any kind: file reads, network calls, database handles, clocks, environment reads. A contract package that touches the network cannot be imported by a test that does not.
- Clients, SDKs, and framework imports: no HTTP client, no ORM base class, no web framework's request model, no task-queue decorator. A framework import re-couples every subsystem to that framework's version.

The line, stated once so it can be quoted in review:

> **The contract layer may define how records resolve; it must never decide an individual case.**

Both halves in one file, so the line is visible rather than theoretical:

```python
# acme/contracts/budget.py

def resolve_budget(requested: int, granted: int, capacity: int) -> int:
    """MECHANISM — belongs here. Pure, total, narrow-only, no preferences.
    Every caller gets the same answer for the same three numbers, forever."""
    return min(requested, granted, capacity)

def pick_default_budget(tenant_tier: str) -> int:
    """POLICY — does not belong here, however small. It encodes what one
    subsystem thinks a tier deserves, unversioned, and every other subsystem
    now inherits that opinion as if it were the shared language."""
    return 100 if tenant_tier == "standard" else 1000
```

The second function is the one that arrives as an innocuous four-line diff, and it is the reason the import rule exists: `pick_default_budget` cannot be written in the contract layer without either hard-coding a subsystem's preferences or importing them. Block the import and the policy has nowhere to land but a versioned policy record.

Serialization libraries are the one nuanced allowance. Depending on a serialization or validation library is defensible if that library *is* the contract's encoding — but it is a dependency the whole repository now shares and pins together, so it is a deliberate, documented choice in `definition-lifecycle.md`, not something that arrives in a diff.

## The gate: enforce it in CI or don't claim it

Direction is a property of the import graph, so check the import graph. `import-linter` reads the AST with `grimp` and needs no runtime import of your code. Two contracts, both earning their place:

```toml
# pyproject.toml
[tool.importlinter]
root_packages = ["acme"]

# 1. Global ordering: each layer may import those below it, never above.
#    Contracts are last, so they may import nothing else in the repository.
[[tool.importlinter.contracts]]
name = "Contracts are the bottom layer"
type = "layers"
layers = [
    "acme.api",
    "acme.scheduler",
    "acme.ingest",
    "acme.contracts",
]

# 2. Belt and braces: contracts may not import any subsystem, stated directly.
[[tool.importlinter.contracts]]
name = "Contracts import no subsystem"
type = "forbidden"
source_modules = ["acme.contracts"]
forbidden_modules = [
    "acme.api",
    "acme.ingest",
    "acme.scheduler",
]
allow_indirect_imports = false   # the default; stated so a later edit can't loosen it
```

Both, not one. The `layers` contract expresses the ordering the whole repository is built on, but its shape is a *list someone edits*: reorder two entries during a refactor and the rule still passes while meaning something else. The `forbidden` contract states the invariant that actually matters, in terms that cannot be satisfied by rearrangement. They also fail independently — an `ignore_imports` exemption added to one contract to unblock a release leaves the other one still failing the build, which is exactly the friction you want at that moment.

Run it as its own CI step:

```yaml
- run: lint-imports --config pyproject.toml   # non-zero exit on any broken contract
```

Alternatives, briefly: a `grimp`-based pytest architecture test asserts the same property inside the suite you already run (useful when you want the failure in the same report as the unit tests, and when the rule needs logic a config file can't express); `tach` enforces module boundaries with per-package manifests and suits repositories that want every package's dependencies declared, not just the contract layer's.

**Rust gets this from the compiler.** Contracts are a leaf crate in the workspace — it declares no path dependencies on subsystem crates, and each subsystem lists `contracts` in its own `[dependencies]`. Cargo forbids dependency cycles between crates outright, so the violation this whole sheet is about is not a lint finding but a build failure with no exemption flag. The lesson transfers: the reason the Python side needs `lint-imports` in CI is that Python's import system will happily do what Cargo refuses to, and a package boundary with no enforcement mechanism is a comment. (See `axiom-rust-workspaces` for crate-graph layout.)

### Adopting the gate on a repository that already violates it

The gate is worth nothing if it is switched on non-blocking "until the cleanup lands". Turn it on blocking on day one, with the existing violations frozen:

1. Run `lint-imports` and capture every reported violation.
2. Add each one — individually, never as a wildcard or a whole-module exemption — to `ignore_imports` on the forbidden contract, with a comment naming the owner and the date it was baselined.
3. The build now passes, and *any new violation fails it*. New violations were always the real risk; the existing ones are a known, bounded, listed quantity.
4. Retire entries as the imports are removed. A list that has not shrunk in two quarters is a finding for the team, not a reason to loosen the rule.

Note the interaction proven above: an `ignore_imports` entry on one contract does not silence the other. Baselining on the `forbidden` contract leaves the `layers` contract failing until the exemption is mirrored there too — deliberate friction, and a second signature on every exemption.

## The leak paths the gate does not close by itself

**`if TYPE_CHECKING:` imports.** `import-linter` counts these as real imports — verified: a `TYPE_CHECKING`-guarded import of a subsystem breaks the contract and fails the build. The leak is not that the tool misses them; it is that the finding gets *exempted*, because "it's erased at runtime, there's no real dependency." There is: the type is still part of the contract's declared interface, the annotation still has to resolve for anyone running a type checker, and the guard is one debugging session away from being promoted to a runtime import by someone who needs the value, not the name. If the contract layer needs to talk about a subsystem's shape, it defines that shape itself as a local `Protocol`:

```python
# acme/contracts/sinks.py — contract-local structural type, no subsystem import
from typing import Protocol

class RecordSink(Protocol):
    """What a subsystem must provide. Defined here; implemented there."""
    def write(self, record: "MetricsRecord") -> None: ...
```

The subsystem's class satisfies it structurally with no registration and no import in this direction. Dependency inversion, in the direction the architecture already requires.

**Transitive leaks.** `contracts` imports `acme.utils`; `acme.utils` imports `acme.ingest`; the import graph is clean at one hop and broken at two. The `forbidden` contract catches this by default — it reports the full chain (`contracts.records -> utils -> ingest`), not just one-hop edges. The reason to write `allow_indirect_imports = false` explicitly anyway is that the loosening is a one-word edit: setting it to `true` makes the two-hop violation pass clean while the direct-import rule still looks enforced, and a flag that was never in the file is much easier to add than one that is in the file, set, and has to be visibly flipped in a diff.

**Convenience re-exports.** `from acme.ingest.types import Collector` in `contracts/__init__.py`, "so consumers have one import." Now every importer of contracts imports `ingest` transitively, the contract package's public surface includes a type it does not own or version, and a change in `ingest` is a breaking change to the shared language. If the type crosses a boundary, it *belongs in contracts* — move the definition, don't forward it.

**Registries and plugin hooks — the one the gate cannot see.** Contracts defines a registry; subsystems register handlers into it at import time; contracts calls those handlers during validation or resolution.

```python
# acme/contracts/registry.py
_VALIDATORS: dict[str, Callable] = {}

def register(kind: str, fn: Callable) -> None: ...
def validate(record) -> None:
    _VALIDATORS[record.kind](record)   # <-- subsystem code, running inside contracts
```

The import graph is spotless. `lint-imports` reports two kept contracts. And the contract layer's behaviour now depends on which subsystems happened to be imported first, its validation outcome varies by process, and subsystem policy is executing inside the shared language exactly as if the import were there — with the added property that the dependency is invisible to every tool you bought to detect it. Callbacks *out* (contracts defines a `Protocol`, subsystems implement it, the caller passes an instance in) are inversion; a registry contracts *calls into* is the same dependency with the arrow drawn in invisible ink. If the gate is your only defence here, you have no defence: this one is caught in review, by asking whether any code path in the contract layer invokes a function it did not define.

## Codegen and IDLs

When contract classes are generated from `.proto` or another IDL, the rule applies to three artifacts identically and for the same reason:

- **The IDL files** live in the contract layer and import only other contract-layer IDL. A `.proto` that imports a subsystem's `.proto` has already lost the direction — the generated code merely reports it.
- **The generated package** is subject to the same lint contract as handwritten contracts. Generated code is not exempt from architecture; it is the code most likely to violate it silently, because nobody reads it in review.
- **The templates and plugin options** are where a violation actually originates. If generated modules import `acme.ingest.helpers`, no amount of editing generated files fixes it — the next regeneration restores the import. **The template is the violation.** Fix it there, regenerate, and commit the generated tree so the lint gate and code review can both see the diff.

## Rationalizations

| Rationalization | Reality |
|---|---|
| "It's one small helper import, not a dependency" | The import graph does not grade imports by size. That one edge makes the subsystem load-bearing for every other subsystem, and it is the precedent every later import cites. Copy the helper in, or move it down into contracts if it is genuinely shared. |
| "Importing the subsystem's enum avoids duplicating it" | If the enum crosses a boundary, it is contract vocabulary and it *belongs in contracts* — move the definition down and have the subsystem import it. Duplication is the wrong fix and upward import is the wrong direction; the third option is the right one. |
| "It's only under `if TYPE_CHECKING`, there's no runtime dependency" | It is still architectural coupling: the contract's declared interface now names a subsystem type, type-checking requires the subsystem present, and the guard gets dropped the first time someone needs the value. Define a contract-local `Protocol` instead. |
| "`utils` is shared infrastructure, it sits below everything" | Then it must import nothing above it, and that must be checked — an unchecked `utils` drifts upward within a quarter and becomes the laundering path that makes a one-hop-clean graph two-hop broken. Give it its own layer entry and `allow_indirect_imports = false`. |
| "The lint rule is there, we just have it non-blocking while we clean up" | A non-blocking gate is a dashboard. Violations accrue at exactly the rate the gate was bought to prevent, and "we'll turn it on after the cleanup" arrives with more violations than it started with. Baseline the existing violations as explicit, individually-listed, dated exemptions and block on everything new. |
| "The registry means contracts doesn't import the subsystem" | It means contracts *executes* the subsystem without importing it. Same coupling, same policy leak, now invisible to the linter. Invert properly: contracts defines the `Protocol`, the caller supplies the implementation. |

## Red flags

- Any `from <subsystem>` or `import <subsystem>` line inside the contract package — including under `if TYPE_CHECKING:`, including in generated files.
- A framework, HTTP client, ORM, or task-queue import in a contract module.
- `contracts/__init__.py` re-exporting a name defined outside the contract package.
- A `utils`, `common`, or `core` package with no layer entry and no direction rule of its own.
- `allow_indirect_imports = true` on the forbidden contract; a growing `ignore_imports` list with no dates and no owner.
- `continue-on-error: true`, `|| true`, or an `allow_failure` marker on the import-lint CI step.
- A module-level mutable registry in the contract layer, or any call in contracts to a function contracts does not define.
- Contracts and one subsystem that must be version-bumped and released together.

## Quick reference

| Situation | Correct move |
|---|---|
| Contracts needs a subsystem's enum | Move the enum definition into contracts; the subsystem imports it |
| Contracts needs to name a subsystem's type | Contract-local `Protocol` + string annotation; never a `TYPE_CHECKING` import |
| Contracts needs subsystem behaviour at runtime | Caller passes an implementation in; never a registry contracts calls into |
| Shared `utils` below contracts | Own layer entry, own direction rule, checked — or fold the helpers into contracts |
| Subsystem-specific default or threshold | Versioned policy record passed as a resolver input, not code in contracts |
| Generated contract package imports subsystem code | Fix the template or plugin options; regenerate; lint the generated package too |
| Existing violations block adopting the gate | Dated, individually-listed exemptions; gate blocks on everything new from day one |
| Rust workspace | Contracts as a leaf crate with no subsystem path dependencies; Cargo rejects cycles at build time |

## Cross-references

- `deterministic-resolution.md` — the mechanism-versus-policy test this sheet enforces structurally; why resolvers take policy as a recorded input.
- `versioned-policy-parameters.md` — where subsystem thresholds and preferences live once they are kept out of the contract layer.
- `definition-lifecycle.md` — review and approval for what enters the contract package, including its third-party dependencies.
- `canonical-identity.md` — canonical serialization and hashing as contract-layer mechanism.
- `silent-default-elimination.md` — validation-at-construction and boundary refusals that belong in contracts.
- `contract-testing.md` — running the import-lint gate alongside contract tests, and testing generated packages.
- Cross-pack: `axiom-rust-workspaces` — leaf-crate layout and workspace dependency-graph discipline.
