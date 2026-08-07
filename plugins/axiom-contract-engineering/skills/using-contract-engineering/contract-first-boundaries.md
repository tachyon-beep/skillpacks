---
name: contract-first-boundaries
description: Use when designing or reviewing what crosses a subsystem boundary — when two subsystems both write the same record class, when a payload field is typed `dict[str, Any]` or `Mapping[str, object]`, when a live mutable object is handed across a boundary, when one record has grown fields for four unrelated consumers, when a nullable field plus a boolean flag can express states the domain forbids, or when review comments are the only thing stopping an illegal record from being emitted. Covers typed immutable records, single producing authority, plain-language role statements, and making illegal states unrepresentable rather than policed.
---

# Contract-First Boundaries

**A subsystem boundary is a set of typed, immutable records, each with exactly one producing authority. Illegal states are unrepresentable in the schema — not caught by a reviewer, not rejected by a downstream validator, not documented in a comment. If a wrong record can be constructed, one eventually will be, and it will be constructed by the subsystem you trusted.**

## When this earns its cost

Read this sheet when:

- You are drawing a new boundary — a new service, a new stage in a pipeline, a new plugin surface — and deciding what shape crosses it.
- Two subsystems both write the same record class, or you are about to let a second one start.
- A record has a `payload: dict[str, Any]`, a `metadata` map, an `extras` blob, or a `context: str` free-text field.
- A consumer is about to add validation logic for a state the producer should never have been able to emit.
- An audit trail is being reconstructed and the records it depends on turn out to have been mutated after they were recorded.
- One record class has accreted fields for four different consumers and every consumer now imports all of it.

The failures this prevents are not parse errors. They are the class where every record is well-formed, every subsystem is individually reasonable, and the *system* holds a state its domain forbids — with no exception, no alert, and a stored history that is no longer evidence of anything.

## The discipline

### 1. Typed, immutable records — never a shared mutable object

What crosses a boundary is a value, not a handle. Concretely: a frozen record whose fields are themselves immutable, serialisable, and typed.

```python
from dataclasses import dataclass
from typing import Mapping

# WRONG — the consumer receives a live handle into the producer's state.
class Order:
    def __init__(self, order_id, lines, status):
        self.order_id, self.lines, self.status = order_id, lines, status

def submit(order: Order) -> None:
    fulfilment.enqueue(order)     # fulfilment now holds the same object
    order.status = "confirmed"    # ...and this mutation is retroactive

# RIGHT — a value, complete at construction, unchangeable afterwards.
@dataclass(frozen=True)
class OrderPlaced:
    order_id: OrderId
    lines: tuple[OrderLine, ...]     # tuple, not list
    placed_at_ms: int
    schema_version: int
```

Why the mutable version is worse than it looks: **it retroactively falsifies audit trails.** Fulfilment logged "processing order X" while X read `status="pending"`. Two milliseconds later the producer flipped it to `"confirmed"`. Every artefact that referenced the object — the log line, the queued job, the in-memory index, the record the auditor will read next quarter — now describes a state that never coexisted with the action taken. Nothing crashed; the history simply became fiction. A recorded value cannot do this, because there is nothing to flip.

Two corollaries that get skipped:

- **Frozen is shallow.** `@dataclass(frozen=True)` with a `list` field is mutable and will be mutated. Use `tuple`, `frozenset`, and nested frozen dataclasses all the way down. A `Mapping` annotation is a promise, not a guarantee — if you must carry one, wrap it (`MappingProxyType`) or convert to a tuple of typed pairs.
- **Amendment is a new record, not an edit.** Correcting a placed order emits `OrderAmended(amends=order_id, ...)`; the original stands. Append-only is what makes "what did we know at the time" answerable. (Immutability is also what makes content-addressing meaningful — `canonical-identity.md`.)

### 2. Exactly one producing authority per record class

For every record class, exactly one subsystem may produce it. Everyone else is a reader. A record class with two writers is two sources of truth wearing one name, and reconciliation is not a feature you can add later — the disagreement has no tiebreaker by construction.

The failure is rarely designed. It arrives as convenience: the evaluation harness emits `EvalResult`, then the retry path also emits one "so the schema stays the same," and now `EvalResult` means either "a model was evaluated" or "an evaluation was abandoned and re-run," and no consumer can tell which. Aggregate metrics double-count and every fix is a heuristic.

Document authority explicitly, per record class, next to the schema:

| Record class | Produced by (sole authority) | Consumed by | Never contains |
|---|---|---|---|
| `EvalRequest` | harness scheduler | model runner | scores, model identity beyond the pinned id |
| `EvalObservation` | model runner | scorer, archive | judgements, pass/fail, thresholds |
| `EvalScore` | scorer | reporter, gate | raw generations, prompt text |
| `EvalGateDecision` | release gate | reporter, humans | anything not in the recorded score set |

The table is the contract artefact, not a diagram annotation. It answers, mechanically: *who do I talk to about a wrong value* (one team, always), *what may I safely depend on* (the fields, not the producer's internals), *what is a breaking change to whom* (the consumer column). Review rule: a diff that adds a second emit site for an existing record class is rejected until either the class is split or the second site is routed through the authority.

When two subsystems genuinely need to say related things, they emit **different record classes** — `ScorerVerdict` and `HumanOverride`, joined downstream — rather than sharing one. The join is then explicit, ordered, and auditable, instead of being a last-write-wins race.

### 3. Every contract carries a plain-language role statement

One sentence, in the schema file, above the type. What this record is, who produces it, and what it must never contain.

```python
@dataclass(frozen=True)
class EvalObservation:
    """What the model actually produced for one request.

    Produced solely by the model runner. It is a record of OBSERVATION, not of
    JUDGEMENT: it must never contain a score, a pass/fail, a threshold, a
    comparison to another model, or any free-text commentary.
    """
    request_id: EvalRequestId
    observation_id: EvalObservationId
    output_tokens: tuple[int, ...]
    latency_ms: int
    runner_version: str
    schema_version: int
```

The forbidden list is not documentation; it is **part of the contract**, and it earns its place because it survives staff turnover. Six months on, someone adds `quick_score: float` "so the reporter doesn't have to re-read the archive," and without the sentence there is no principle to point at — only taste, which loses arguments to deadlines.

The strongest form of the forbidden list is one that cannot be violated: **forbidden fields are schema-invalid, not merely discouraged.** A judgement field on an observation record should fail to construct, fail to parse, and fail a test that asserts the forbidden name never appears in the serialised form. See `blinding-by-construction.md` for building input schemas that structurally cannot carry the influence you are trying to exclude; this sheet's role statement is the human-readable half of that discipline, and the tests in `contract-testing.md` are the enforcing half.

### 4. Make illegal states unrepresentable

Four mechanisms, in rough order of how much they buy:

**Closed enums, never strings.** A `str` field admits every typo, every alias, and every future value nobody agreed to. An enum admits what the contract declares — and adding a member is a visible, versionable diff.

```python
class RejectionReason(str, Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_CANDIDATE_ELIGIBLE = "no_candidate_eligible"
    POLICY_BLOCKED = "policy_blocked"
```

**Tagged unions, never nullable-field pairs.** This is where most contradictory states live:

```python
# WRONG — the type admits four states; the domain permits two.
@dataclass(frozen=True)
class Selection:
    selected_id: str | None
    rejected: bool
    rejection_reason: str | None
    # (id=X, rejected=True)  -> selected AND rejected. Contradiction.
    # (id=None, rejected=False) -> neither. Contradiction.
    # Consumers now branch on combinations, each guessing differently.

# RIGHT — the union has exactly the two members the domain has.
@dataclass(frozen=True)
class Selected:
    candidate_id: CandidateId

@dataclass(frozen=True)
class RejectedAll:
    reason: RejectionReason

Selection = Selected | RejectedAll
```

Nullable-pair schemas do not merely permit contradictions — they *distribute the reconciliation*. Every consumer invents its own precedence rule (`if rejected: ... elif selected_id: ...` vs the reverse), the rules disagree, and the disagreement is invisible until two reports differ. The union removes the question.

This is a different job from the `Measured | Absent` union in `silent-default-elimination.md`: that one encodes **absence of information**; this one encodes **mutual exclusion of outcomes**. Same mechanism, different defect — a record can need both.

**Validate at construction.** An invalid record should not exist, even transiently, even inside the producer, even in a test fixture:

```python
@dataclass(frozen=True)
class EvalScore:
    request_id: EvalRequestId
    observation_id: EvalObservationId
    value: float
    scorer_version: str
    schema_version: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ContractViolation(f"score out of range: {self.value}")
        if self.scorer_version == "":
            raise ContractViolation("scorer_version must be non-empty")
```

Parser-side rejection (`silent-default-elimination.md`) guards records arriving from *outside*. `__post_init__` guards records your own code builds and never sends over a wire. Both layers are needed: most illegal records in practice are constructed internally by a producer that "knows" its own data.

**Distinct ID types.** `NewType` (or a frozen single-field wrapper) stops the swap the type checker would otherwise wave through:

```python
from typing import NewType
EvalRequestId = NewType("EvalRequestId", str)
EvalObservationId = NewType("EvalObservationId", str)
CandidateId = NewType("CandidateId", str)
# score_for(request_id=obs.observation_id)  -> type error, not a silent mis-join
```

Everything is a `str` at runtime, so the cost is zero and the catch is at review time — the point in the lifecycle where an argument-order mistake is free to fix.

### 5. Distinct concerns stay distinct records

An **observation** (what happened), a **commission** (what a subsystem was asked to do), the **constraints** in force, and the **context** a decision was made under are four different facts with four different producers, four different lifetimes, and four different forbidden lists. Merge them and each becomes an undocumented side channel for the others: constraint fields ride along on the observation, so the scorer can read the budget it was supposed to be blind to; commission fields ride on the constraint record, so the runner learns what answer was hoped for.

The **god-record** anti-pattern is the end state. One class accretes every field any consumer ever wanted, because adding a field is easier than adding a record class. The costs compound:

- Every consumer is coupled to all of it. A field added for the reporter is a breaking-change surface for the gate.
- The forbidden list becomes unwritable — the record legitimately contains judgements *and* observations, so "must never contain a judgement" cannot be said about it.
- Authority collapses (§2): a record with fields from four subsystems will end up written by four subsystems.
- Versioning becomes maximally expensive: any change to any concern bumps the version every consumer pins (`schema-versioning-and-evolution.md`).

The related smell is the **dict-of-anything payload field** — `payload: dict[str, Any]`, `metadata: Mapping[str, object]`, `extras`. It is a god-record with the type checker switched off: unenumerable keys, no forbidden list, no evolution story, and no way to test that a forbidden value never appears. If some fields are genuinely open-ended, that is a signal you have found a *second* record class, not a reason to open a hole in the first. The narrow legitimate case — opaque bytes forwarded verbatim and never read by this subsystem — is typed as `bytes` with a declared owner, not as a map anyone may reach into.

The corrective when you find a god-record: split by *producer*, then by *forbidden list*. Fields whose sole authority differs must be different records. Fields that must be invisible to a given consumer must be different records, so that invisibility is achieved by not passing the record rather than by asking the consumer not to look.

### 6. Records bind identity

Split records only pay off if the pieces can be rejoined. Each derived record carries **explicit reference fields** naming the records it derives from:

```python
@dataclass(frozen=True)
class EvalGateDecision:
    decision_id: DecisionId          # this record's own instance identity
    run_id: RunId                    # correlation identity: the whole eval run
    scored: tuple[EvalScoreId, ...]  # exactly which scores were consumed
    outcome: GateOutcome             # closed enum
    policy_version: str
    schema_version: int
```

Two identity fields with two distinct jobs, and conflating them is a real defect:

- **Record-instance identity** (`decision_id`) names *this record*. It is unique per record, it is what other records reference, and it is what deduplication and content-addressing key on.
- **Correlation identity** (`run_id`) names *the activity* the record belongs to. Many records share it. It is what you filter a log by, and what joins the chain end to end.

Using one field for both breaks in both directions: a retry that reuses the instance id creates two different records claiming to be the same one (dedup silently drops the second); a per-record correlation id makes the run unreconstructible. Name both fields, in every record, and let the reference fields — not timestamp proximity, not log adjacency — carry the chain.

Rules that keep the chain sound: references are to *specific* records (`scored: tuple[EvalScoreId, ...]`, not `score_count: int`); a reference is never inferred from ordering or arrival time; and derived records also carry the versions of what they consumed (`schema-versioning-and-evolution.md`, `versioned-policy-parameters.md`) so an archived decision can be interpreted under the schema and policy that actually produced it.

## Rationalizations

| Rationalization | Reality |
|---|---|
| "The consumer will validate it" | Then the illegal state is representable, transmissible, and storable — validation only decides whether *this* consumer notices. Every other consumer, the archive, and the replay path still receive it. Unrepresentable beats validated. |
| "Both services need to write it, they just write different fields" | Then they are two record classes that happen to share a name. Split them and join downstream; there is no tiebreaker for a field two authorities disagree about. |
| "We pass the object by reference for performance" | You bought a copy and sold your audit trail. If profiling actually shows record construction on the hot path, that is an internal-representation problem inside one subsystem — boundaries still get values. |
| "A `dict` payload keeps us flexible for future fields" | It keeps you flexible about *meaning*, which is the one thing a contract exists to fix. Untyped keys cannot be versioned, forbidden, or tested. Add a record class; that is what flexibility looks like when it is safe. |
| "Nullable fields are simpler than a union" | Simpler to *write*, and then every consumer independently invents a precedence rule for the contradictory combinations. The union is where the branching stops. |
| "A comment says not to set both" | A comment is a request. `__post_init__` is a contract, and a schema that cannot express the state is a guarantee. Prefer in that order. |
| "It's one extra field, splitting the record is overkill" | The field arrives with a different producer or a different forbidden list — that is the split criterion, not field count. God-records are built one reasonable field at a time. |
| "We'll add the reference id when we need to trace it" | Tracing is needed after an incident, on records already written. The chain must be built before it is queried; retrofitting it means guessing joins from timestamps. |

## Red flags

- A record class with more than one emit site, or a `dict[str, Any]` / `Mapping[str, object]` / `metadata` / `extras` field crossing a boundary.
- `@dataclass(frozen=True)` containing a `list`, `dict`, or `set`; or a boundary type with setters, `@property` setters, or any in-place mutation method.
- A nullable field paired with a boolean or a second nullable that can express states the domain forbids.
- Bare `str` where the value comes from a fixed set; bare `str` for IDs of more than one record class in the same function signature.
- A schema file whose types have no docstring naming producer and forbidden contents.
- A record class every subsystem imports; a version bump that forces every consumer to re-pin regardless of what changed.
- Downstream code that reconciles fields from the same record ("if both set, prefer…").
- Derived records with no reference field to their inputs, or a single `id` doing both correlation and instance duty.

## Quick reference

| Concern | Correct shape |
|---|---|
| What crosses a boundary | Frozen dataclass, immutable all the way down; amendment emits a new record |
| Who may produce a record class | Exactly one subsystem; documented in a producer/consumer table per class |
| Two subsystems with related things to say | Two record classes, joined downstream by explicit reference |
| What a record must never contain | Named in the role-statement docstring; enforced as schema-invalid + a test |
| Value from a fixed set | Closed enum, never `str` |
| Mutually exclusive outcomes | Tagged union of the legal members, never nullable-field pairs |
| Field invariants (ranges, non-empty, cross-field) | `__post_init__` raising `ContractViolation` — invalid records never exist |
| IDs of different record classes | Distinct `NewType`s so a swap is a type error |
| Open-ended extra fields | A second record class — never an untyped map |
| Provenance | Explicit reference fields to source records; instance id and correlation id as separate fields |

## Cross-references

- `blinding-by-construction.md` — making a forbidden field structurally impossible rather than merely listed; the enforcing half of §3.
- `silent-default-elimination.md` — the absence-encoding union (a different job from §4's outcome union) and parser-side rejection, which layers with `__post_init__`.
- `schema-versioning-and-evolution.md` — what changing one of these records costs, and why god-records make versioning maximally expensive.
- `canonical-identity.md` — content-addressing the immutable records this sheet defines; canonical form of reference fields.
- `versioned-policy-parameters.md` — binding the policy version onto derived records so the chain in §6 stays interpretable.
- `deterministic-resolution.md` — how a derived record is computed from the records it references, reproducibly.
- `contract-testing.md` — forbidden-field tests, illegal-state-construction tests, and producer-authority checks.
- `definition-lifecycle.md` — how a new record class and its authority assignment get approved before anything emits it.
