---
name: static-vs-runtime-tradeoffs
description: Use when deciding whether an invariant should be enforced statically (analyzer rule), at runtime (assertion / contract), or both — and to write down the boundary explicitly so future contributors do not silently move checks between the two. Covers what static analysis can and cannot decide (the Rice-theorem ceiling), the dual-enforcement pattern (decorator-as-assertion), the cost model (developer time, build time, runtime overhead, blast radius), and the discipline of keeping the boundary statement testable. Produces `06-static-runtime-boundary.md`.
---

# Static vs Runtime Tradeoffs

## Why the Boundary Is Worth Writing Down

Every invariant a system depends on is enforced by *something*: a static analyzer rule, a runtime assertion, a type system, a code review checklist, or — most often — wishful thinking. Without an explicit boundary statement, the same invariant gets re-enforced in three places (waste) or in zero places (silent failure). When an incident happens, the team adds a check wherever the bug landed; over time, the check distribution looks random.

`06-static-runtime-boundary.md` exists to make the choice explicit per invariant. The output is a small ledger:

| Invariant | Where enforced | Rationale |
|-----------|----------------|-----------|
| Untrusted input never reaches `os.system` | static (rule STA001) | tractable; security-bearing; prefer prevention |
| Database connection limit ≤ 50 | runtime (config validator + connection pool) | depends on runtime config; static can't see it |
| `User.email` is a non-empty string | static (type) + runtime (Pydantic validator) | dual enforcement; type catches construction sites; validator catches deserialisation |

The discipline: **every invariant has at least one named enforcer, and the choice is justified.**

## What Static Analysis Can Decide

Static analysis decides properties of the *program text* — what the code says, structurally and (with type information) semantically. The well-served categories:

- **Structural invariants** — "no function has more than 30 parameters", "no module imports both X and Y", "every public function has a docstring". The AST tells you directly.
- **Type properties** — "this argument is annotated as `str` and is called with an `int` somewhere". A type system (or a typed dataflow analyzer) decides this.
- **Reachability and flow** — "this `return` is unreachable", "this variable may be used before assignment". Phase 1 of `three-phase-inference.md`.
- **Call-and-flow patterns** — "untrusted input flows to a sink without sanitisation", "this resource is opened and not closed on every path". Phase 3 with a lattice from `taint-lattice-design.md`.
- **Symbolic constants** — "this regex is malformed", "this format string has the wrong arity". The text contains the answer.

If the answer to "is this property satisfied?" reduces to "look at the code structure and types," static analysis is the right tool.

## What Static Analysis Cannot Decide (the Rice Ceiling)

By Rice's theorem, no non-trivial *semantic* property of programs is decidable in general. In practice, this means static analyzers approximate. The categories where the approximation usually fails:

- **Properties of runtime values** — "this string is a valid SQL identifier," "this integer is in the range the database accepts," "this URL points to an allowed host." The value is unknown until runtime; static can only reason about *types*, not *values*.
- **Properties of the environment** — "the configuration loaded at startup has the database URL set," "the file at `/etc/config.toml` exists and is readable." The environment is unknown to the analyzer.
- **Properties of dynamic dispatch** — "every implementation of this method respects the contract." Static can check declared types; it cannot run the method to check the contract holds.
- **Concurrent and ordering properties** — "this code is free of data races," "this lock is always released." Some analyzers handle some of this; full coverage is research-grade and brittle.
- **Cross-process and cross-network properties** — "the message we send matches the schema the receiver expects." Static within a process; the wire is opaque without contracts (which is *why* contract-first design exists).

When the property is in this list, the answer is runtime enforcement. The static rule, if it exists at all, can only enforce the *structural prerequisite* (e.g., "all values reaching the SQL execute call are passed through `sql_param_bind`"; the parameter binding then decides validity at runtime).

## The Dual-Enforcement Pattern

Many invariants are best served by *both* static and runtime. Three common shapes:

### Type-and-validate

The type system asserts the field is `str`; the validator asserts it is a non-empty string conforming to RFC 5321. Static catches construction sites (a `User(email=42)` fails the type checker); runtime catches deserialisation (a JSON `{"email": 42}` fails the validator).

```python
@dataclass(frozen=True)
class User:
    email: str   # static: type checker enforces
    # runtime: validator at construction enforces RFC 5321
    def __post_init__(self) -> None:
        if not is_valid_email(self.email):
            raise ValueError(...)
```

### Decorator-as-assertion

A single decorator drives both sides. The decorator wraps the function for runtime enforcement; the analyzer reads the decorator at static time to enforce the same property structurally.

```python
@requires_permission("admin")  # runtime: checks at call; static: rule reads decorator
def delete_user(user_id: UserId) -> None: ...
```

The static rule reads the decorator and verifies that callers either are themselves `@requires_permission("admin")` or have a check earlier in the call chain. The runtime check catches the residual (a caller invoked from a context the static analysis couldn't reach — eval, dynamic dispatch, third-party callback). v0.2.0 sheet `decorator-as-assertion.md` covers this pattern in depth, including how to verify the static and runtime models agree.

### Static guard + runtime fallback

The static rule catches the bulk of the cases; a runtime fallback exists for the residual and logs whenever it fires. The log entries become the test corpus for refining the static rule.

```python
def execute_query(sql: str, params: tuple) -> Cursor:
    if not is_param_bound(sql):  # runtime fallback
        log.error("Static analyzer should have caught this", extra={"sql": sql})
        raise UnsafeQuery(...)
    return cursor.execute(sql, params)
```

Over time, the runtime log entries should approach zero. If they don't, the static rule has a known gap and the runtime check is doing the work it was supposed to enforce.

## The Cost Model

Static and runtime checks have different costs paid in different places.

| Cost | Static | Runtime |
|------|--------|---------|
| **Developer time** to write | Higher (rule, lattice, tests) | Lower (one assertion line) |
| **Build / CI time** | Pays at every commit | None |
| **Runtime overhead** | Zero | Per-call cost |
| **False positive cost** | Developer attention; suppression accumulation | None (it didn't fire) |
| **False negative cost** | Bug ships | Bug detected at runtime; may be too late |
| **Blast radius of a fix** | One rule update fixes all callers | Each caller may need separate update |
| **Visibility** | Findings in the report; rule documented | Stack trace in production; documented in code |
| **Auditability** | Rule + waivers (see `false-positive-economics.md`) | Logs (if logged) |

**The decision heuristic:**

- **Prefer static** for security-bearing invariants where a runtime catch is too late (injection, auth bypass, privilege escalation).
- **Prefer runtime** for invariants depending on runtime values (input validation, range checks, environment).
- **Use both** for invariants that are both: structural in some part, value-dependent in another. Pay the dual-enforcement cost; document the agreement contract.
- **Avoid both** when the static rule is so coarse that the runtime check does all the work — the static rule is then ceremony. Either refine the lattice or retire the static rule.

## The Boundary Statement

`06-` produces a ledger with three columns and a justification per row. The ledger is the artifact other contributors consult before adding a new check. A useful template:

```
Invariant: [precise statement, testable]
Where enforced:
  - static: [rule id or "none"]
  - runtime: [assertion location or "none"]
  - dual: [if both, agreement contract: how do we verify static and runtime
           do not drift]
Justification:
  - tractability: [decidable statically? values vs types? Rice-relevant?]
  - cost: [build-time vs runtime; FP cost; blast radius]
  - severity: [if missed, what breaks?]
Boundary owner: [who decides if this moves]
Last reviewed: [date]
```

For the consistency gate (check 9): a developer reading `06-` should be able to correctly classify a *new* invariant. If the ledger doesn't generalise — if it's an arbitrary list — the boundary isn't articulated, only enumerated.

## When to Move a Check

Moving a check from runtime to static (or vice versa) is a deliberate act, not a drift. The move triggers a re-emission of `06-` plus, depending on direction:

| Direction | Required updates |
|-----------|-------------------|
| runtime → static | Add the rule (`04-`); extend the test corpus; FP-rate baseline reset (`05-`); plan removal of the runtime check after grace period |
| static → runtime | Deprecate the rule (`04-` deprecation flow); add the runtime check; document the rationale (usually: rule unsoundness or excessive FP rate) |
| add dual enforcement | Add the second side; write the agreement contract; pay the dual cost knowingly |
| remove dual enforcement | Confirm the remaining side actually catches what the removed side caught; update the ledger |

The move is **not** "we noticed a bug and added a check wherever the bug landed." That is drift. Drift produces the random check distribution this sheet exists to prevent.

## The Anti-Pattern: "We'll Make It Static Later"

A common pattern: the team writes runtime checks now ("for safety"), with the explicit plan to lift them to static rules later ("when we have time"). Three years later: zero static rules, hundreds of runtime checks, no plan.

The anti-pattern's tells:

- TODO comments referring to a future static analyzer that does not exist.
- Runtime checks with the word "temporary" in their docstrings.
- A backlog of "make X static" tickets with no ETA.

The fix is honesty in `06-`: if the long-term home is static, plan it; if it isn't, drop the pretense and accept the runtime cost. The analyzer that doesn't exist isn't enforcing anything.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Static rule for a value-dependent property (e.g., "this string is a valid email") | High FP rate; lattice grows special cases | Move to runtime; document in `06-` |
| Runtime check for a structural property the type system already enforces | Redundant cost; type-shaped runtime errors | Remove the runtime check; trust the type |
| Dual enforcement without agreement contract | Static and runtime drift; static evolves, runtime stays, gap appears | Write the agreement contract; v0.2.0 sheet `decorator-as-assertion.md` covers the verification pattern |
| "We'll make it static later" with no plan | Static layer never materialises; runtime cost permanent | Drop the pretense or commit a date |
| Ad-hoc check addition wherever a bug landed | Random check distribution; ledger out of date | Re-emit `06-` after every check addition; treat the ledger as authoritative |
| Boundary statement is "case-by-case" | Not a boundary; check distribution stays random | The ledger must generalise; if it can't, articulate the *principle* and apply it |
| Static rule retired without removing the runtime fallback | Code still pays runtime cost for caught-by-static cases | Retire both sides together; document |

## The Decision Output (`06-static-runtime-boundary.md`)

A complete `06-` answers:

1. **The invariant ledger** — every load-bearing invariant, where it is enforced, the rationale.
2. **The decision principle** — the rule by which a *new* invariant is classified; the ledger should follow from the principle, not be arbitrary.
3. **Dual-enforcement contracts** — for every invariant in the dual column, the agreement statement and how it is tested.
4. **Move discipline** — what triggers a check moving between layers; what re-emission cascade follows.
5. **Cost model snapshot** — current static cost (build time, FP rate), current runtime cost (per-call overhead, log volume).
6. **Anti-pattern audit** — current count of TODO-future-static comments, "temporary" runtime checks, deferred backlog tickets; trend over time.
7. **Boundary owners** — who has authority to move a check; who reviews the ledger; cadence.
8. **Out-of-scope invariants** — invariants the system depends on but neither static nor runtime enforces (e.g., "the deployment script is run from main"); how those are governed (process, code review, runbook).

## Cross-References

- `taint-lattice-design.md` — what the lattice can express; properties outside the expressivity of the lattice fall to runtime
- `three-phase-inference.md` — over-approximation in inference is a static-vs-runtime question: do we accept the FPs or move to runtime?
- `plugin-architecture-for-analyzer-rules.md` — when a rule retires, this sheet records the runtime check that replaces (or doesn't replace) it
- `false-positive-economics.md` — the cost of runtime is per call; the cost of static is per FP; this sheet weighs them
- v0.2.0 planned: `decorator-as-assertion.md` — the descriptor pattern for dual enforcement with verified agreement
- Cross-pack: `axiom-audit-pipelines:audit-aware-logging-vs-observability` — the analogous "what's evidence-grade vs ordinary" boundary, same discipline applied to logs
- Cross-pack: `ordis-security-architect:design-controls` — for security-bearing invariants, the threat model dictates whether prevention (static) or detection (runtime) is the right control
- Cross-pack: `axiom-sdlc-engineering:quality-assurance` — verification (does the check work) vs validation (is the check the right one) discipline applied to the boundary itself
