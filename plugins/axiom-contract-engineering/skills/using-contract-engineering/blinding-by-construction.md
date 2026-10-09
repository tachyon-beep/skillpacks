---
name: blinding-by-construction
description: "Use when a consumer must not see some field a producer holds \u2014 a blinded evaluation or QA harness that must not learn which vendor produced a candidate, a reviewer service that must not see a regulated provider field, an adjudicator that must not learn who submitted what \u2014 or when someone proposes redacting a value at serialization, popping a key from a dict copy, marking a field \"deprecated, nobody reads it\", or passing a free-text note beside carefully typed fields. Covers separate view types, allowlist projection, field-policy closure, canary tests, covert channels, and the producer-side mirror (fields a producer must not be able to say)."
---

# Blinding by Construction

**If a consumer must not see a field, the field is absent from that consumer's schema — a separate view type that has no such field — never present-and-ignored, present-and-redacted, or excluded by convention. Blinding by ignoring is not blinding; it is a promise that every future edit is free to break.**

## When this earns its cost

Read this sheet when:

- A stage of a pipeline must decide without knowing something the record upstream carries: a blinded evaluation harness that must not learn which vendor produced a candidate; a reviewer service that must not see a regulated provider field; an adjudicator that must not know who submitted what.
- A record class is described as "we just don't read that field downstream", "it's deprecated", or "the consumer ignores it".
- Someone proposes redaction at serialization — `out["provider"] = "<redacted>"` on a dict copy — or a projection built by copying everything and deleting the sensitive keys.
- A blinded or authority-constrained record has a `notes`, `detail`, `context`, or `rationale` string on it.
- The mirror-image case: a *producer* must not be able to express a preference — no topology hint, no diagnosis, no "I'd like outcome X" — and the plan is to ignore it if present.

The failure is quiet by construction. A leak through a blinded boundary produces no error, no latency change, and no log line; it produces *slightly better-looking results* and nobody can say why. The evaluation that was supposed to be blind was merely polite about it.

## The defect class

Every one of these is the same defect — the field is *present* at the boundary and something downstream is trusted not to use it:

```python
# WRONG — redaction at serialization. The field exists; one code path stamps
# over the value. Any other serializer, log line, or debug dump still has it.
out = dataclasses.asdict(record)
out["provider"] = "<redacted>"

# WRONG — deny-list projection. Fails open: every field added to
# CandidateRecord after today is exposed by default, silently, forever.
out = dataclasses.asdict(record)
del out["provider"]

# WRONG — blinding by convention. Enforced by a comment.
@dataclass
class CandidateRecord:
    provider: str   # reviewers MUST NOT read this

# WRONG — blinding by omission-in-practice. The reviewer gets the whole
# record and happens not to touch .provider today.
def review(record: CandidateRecord) -> Decision: ...
```

The shared property: **the secret is inside the reviewer's blast radius**, and staying blind depends on the continued good behavior of code nobody is currently writing. A deny-list is the worst of them because it looks like a control. It is a control over the fields that existed when someone wrote it.

## The discipline

### 1. Separate view types, built by allowlist

The consumer's schema is a *different type* with no sensitive field on it. Not the source type with a flag, not the source type with `Optional[str] = None`, not a subclass. A different class, whose complete field list is the reviewer's entire world.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ArtifactRecord:
    body: str
    toolchain_fingerprint: str        # identifies the vendor's build — sensitive

@dataclass(frozen=True)
class CandidateRecord:               # the system of record; holds everything
    candidate_id: str
    artifact: ArtifactRecord
    provider: str
    submitted_at: float
    schema_version: int

@dataclass(frozen=True)
class ReviewerCandidateView:
    """The reviewer's whole world. There is no `provider` field to forget to
    strip, and no `asdict()` firehose that could carry one."""
    candidate_id: str
    body: str
    schema_version: int

REVIEWER_VIEW_VERSION = 1

def project_for_reviewer(rec: CandidateRecord) -> ReviewerCandidateView:
    # Allowlist. Every exposed field is named on its own line. A field added
    # to CandidateRecord tomorrow does not appear here until someone writes
    # the line — which is exactly the review conversation you want to force.
    return ReviewerCandidateView(
        candidate_id=rec.candidate_id,
        body=rec.artifact.body,
        schema_version=REVIEWER_VIEW_VERSION,
    )
```

Two rules make this real rather than aesthetic:

- **The projection is the only constructor of the view.** If the reviewer can be handed a `CandidateRecord` on any code path — a test helper, a debug endpoint, a retry that re-reads from the store — the type separation buys nothing.
- **No generic serializer crosses the boundary.** `asdict()`, `__dict__`, `vars()`, `model_dump()`, and pickle are firehoses: they emit whatever the object has. The view exists precisely so that a firehose over *it* is safe.

The view is a contract record like any other: it carries its own `schema_version`, and changing what it exposes is a version bump (`schema-versioning-and-evolution.md`) — widening a blinded view is a semantic change, not a convenience.

### 2. Three layers, in increasing order of surviving a careless change

**(a) The type has no field.** §1. Cheap, and it stops the accident where someone reaches for `record.provider` because autocomplete offered it. It does not stop someone adding `provider` to the view "just for debugging".

**(b) A field-policy registry with a closure walk.** Every field of every dataclass *reachable* from the source record must carry an explicit `EXPOSE` or `REDACT` decision. Reachability matters: nesting is where leaks hide — nobody adds `provider` to the top-level view, but somebody adds `toolchain_fingerprint` to `ArtifactRecord`, three levels down, and it rides out inside an already-approved sub-record.

```python
from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any, get_args, get_origin


class FieldPolicy(str, enum.Enum):
    EXPOSE = "expose"
    REDACT = "redact"


class UndeclaredFieldPolicy(Exception):
    """A field reachable from a blinded source carries no EXPOSE/REDACT
    decision. Raised at import time and asserted in CI."""


# (qualified class name, field name) -> decision. There is no default.
FIELD_POLICY: dict[tuple[str, str], FieldPolicy] = {
    ("CandidateRecord", "candidate_id"):        FieldPolicy.EXPOSE,
    ("CandidateRecord", "artifact"):            FieldPolicy.EXPOSE,
    ("CandidateRecord", "provider"):            FieldPolicy.REDACT,
    ("CandidateRecord", "submitted_at"):        FieldPolicy.REDACT,
    ("CandidateRecord", "schema_version"):      FieldPolicy.EXPOSE,
    ("ArtifactRecord", "body"):                 FieldPolicy.EXPOSE,
    ("ArtifactRecord", "toolchain_fingerprint"): FieldPolicy.REDACT,
}


def _walk_type(tp: Any, seen: set[type], missing: list[str]) -> None:
    """Descend one annotation. Containers are unwrapped to their arguments;
    dataclasses recurse; everything else is a leaf with no fields to decide."""
    if get_origin(tp) is not None:
        # Optional[X] is Union[X, None]; also covers list[X], tuple[X, ...],
        # Mapping[K, V] (both K and V are walked — a mapping keyed by a
        # dataclass hides fields just as well as one valued by it).
        for arg in get_args(tp):
            if arg is not Ellipsis:
                _walk_type(arg, seen, missing)
        return
    if dataclasses.is_dataclass(tp):
        _walk_dataclass(tp, seen, missing)
    # str / int / float / bool / Enum / None terminate the walk.


def _walk_dataclass(cls: type, seen: set[type], missing: list[str]) -> None:
    if cls in seen:                      # self-referential records must not loop
        return
    seen.add(cls)
    # get_type_hints, NOT dataclasses.Field.type: under
    # `from __future__ import annotations` the latter is the *string*
    # "ArtifactRecord", get_origin/is_dataclass both say no, and the walk
    # stops silently at the first nested record — passing while checking
    # nothing. This one line is the difference between a control and a
    # decoration.
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if (cls.__qualname__, f.name) not in FIELD_POLICY:
            missing.append(f"{cls.__qualname__}.{f.name}")
        _walk_type(hints[f.name], seen, missing)


def assert_field_policy_closure(*roots: type) -> None:
    missing: list[str] = []
    seen: set[type] = set()
    for root in roots:
        _walk_dataclass(root, seen, missing)
    if missing:
        raise UndeclaredFieldPolicy(
            "no EXPOSE/REDACT decision for: " + ", ".join(sorted(missing))
        )


# Import-time enforcement. Adding a field anywhere in the reachable graph
# breaks the import until a human decides what the blinded consumer may see.
# CI additionally asserts this for every blinded root (contract-testing.md).
assert_field_policy_closure(CandidateRecord)
```

The registry does not *do* the blinding — the view type does. The registry makes the omission of a decision impossible to commit. Note the direction of failure: an undeclared field fails **closed** at import, not open at runtime.

**(c) A canary test with a positive control.** Plant a sentinel *value* in the sensitive field and assert it appears nowhere in the serialized blinded output. Assert separately that it *is* present in the source record — otherwise a projection that returns an empty view, or a fixture builder that quietly stopped setting `provider`, passes forever while checking nothing.

```python
SENTINEL = "canary-provider-8f31c2ae"

def test_provider_cannot_reach_the_reviewer() -> None:
    rec = make_candidate(provider=SENTINEL, toolchain_fingerprint=SENTINEL)

    # Positive control: the sentinel really is in the source. Without this,
    # an empty or broken projection passes vacuously.
    assert SENTINEL in json.dumps(dataclasses.asdict(rec))

    blob = canonical_dumps(project_for_reviewer(rec))

    assert SENTINEL not in blob                       # the blinding claim
    assert json.loads(blob)["candidate_id"] == rec.candidate_id   # non-vacuous
```

Run the sentinel through *everything the reviewer can observe*, not just the happy-path payload: the view, the error messages the reviewer receives, the correlation ID, any metrics labels or trace attributes emitted on its behalf. A provider name that leaks through an exception string is as leaked as one in the body.

### 3. Forbidden fields are schema-invalid, not discouraged

The mirror case: an upstream record must never carry a particular *influence* — a topology hint, a diagnosis, a preferred outcome. Two things must be true, and the second is the one teams skip:

1. The schema has no such field.
2. The parser **rejects** payloads that contain it.

Rejecting rather than ignoring is the whole point. If an unknown `preferred_outcome` key is silently dropped, a producer can keep emitting influence indefinitely and nobody ever learns it tried. Rejection converts an attempt into an incident with a producer's name on it.

```python
# Named, closed set of keys that are schema-invalid at this boundary. This is
# not the same as unknown-field handling: unknown additive fields may be
# preserved-and-ignored if the contract's evolution rules say so
# (schema-versioning-and-evolution.md). These specific names never may.
FORBIDDEN_KEYS = frozenset({"provider", "preferred_outcome", "diagnosis_hint"})

def parse_reviewer_submission(raw: Mapping[str, Any]) -> ReviewerDecisionRecord:
    intruders = FORBIDDEN_KEYS & raw.keys()
    if intruders:
        raise ContractViolation(
            f"schema-invalid fields present at blinded boundary: {sorted(intruders)}"
        )
    ...
```

An **authority test** proves the forbidden field cannot arrive: construct a payload containing it and assert the parse raises, for every forbidden name and every schema version the parser accepts (`contract-testing.md`). Without that test, "the schema has no such field" is a claim about a dataclass, not about the boundary.

### 4. Free text is a covert channel

A `notes: str`, `detail: str`, `context: str`, or `rationale: str` crossing a blinded or authority-constrained boundary defeats every typed control above. The vendor name appears in the note. The diagnosis appears in the rationale. The preferred outcome appears as "note that option B has worked well historically". The typed fields were blinded with care and the prose field carries everything they excluded — in a field nobody grep'd, because it is *supposed* to hold arbitrary text.

The rule, and it is absolute at these boundaries:

- **Reasons are closed enums.** A fixed, versioned set of values. Adding a reason is a schema change with a review, which is exactly the friction that keeps the channel closed. (This is the rule `silent-default-elimination.md` relies on for its `AbsenceReason` type — an absence reason with a free-text `detail` field is the same defect.)
- **Parameters are numeric-only maps**, keyed by a closed enum, with the numeric type enforced at projection time — not merely annotated.
- **Human-facing prose lives outside the channel**, in the un-blinded review surface, keyed by correlation ID. Reviewers still write paragraphs; those paragraphs simply do not travel with the record that crosses the boundary.

```python
class DeclineReason(str, enum.Enum):
    FAILS_ACCEPTANCE = "fails_acceptance"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"

class ScoreAxis(str, enum.Enum):
    CORRECTNESS = "correctness"
    CLARITY = "clarity"

@dataclass(frozen=True)
class ReviewerDecisionRecord:
    correlation_id: str
    outcome: Outcome
    reasons: tuple[DeclineReason, ...]        # closed enum — no free text
    scores: Mapping[ScoreAxis, float]         # numeric only
    schema_version: int
    # Deliberately absent: notes / detail / rationale. Prose belongs to the
    # un-blinded review UI, joined by correlation_id after adjudication.

def project_decision(raw_scores: Mapping[ScoreAxis, Any]) -> Mapping[ScoreAxis, float]:
    for axis, value in raw_scores.items():
        if isinstance(value, str) or not isinstance(value, (int, float)):
            raise ContractViolation(f"non-numeric score for {axis}: {value!r}")
    return {axis: float(v) for axis, v in raw_scores.items()}
```

The same closure applies to resolvers: no free text reaches a resolver's input schema, because a passthrough string is an unversioned preference channel (`deterministic-resolution.md`). And the identifiers themselves count as text — a `correlation_id` of `"acme-corp-0042"` blinds nothing. Correlation IDs at a blinded boundary are opaque and content-independent; ordering and batch position are covert channels too, so the blinded batch is shuffled or canonically ordered by the opaque ID.

### 5. Provenance round-trip: removed from the view, not from the system

Blinding is not amnesia. The pipeline still needs to know which vendor produced the candidate the reviewer just scored — afterwards. So provenance lives **outside** the blinded view, in the system of record, keyed by the correlation ID, and is reattached only once the blinded stage has completed.

```python
@dataclass(frozen=True)
class ProvenanceEntry:
    correlation_id: str
    provider: str
    submitted_at: float
    source_schema_version: int

def rejoin(
    decisions: Sequence[ReviewerDecisionRecord],
    ledger: Mapping[str, ProvenanceEntry],
) -> list[AttributedDecision]:
    """Called after adjudication closes — never during, and never by any
    component that also holds a reviewer's view."""
    out = []
    for d in decisions:
        entry = ledger.get(d.correlation_id)
        if entry is None:
            # A decision with no provenance is unattributable forever. Fail
            # loudly here; do not drop it and do not invent an entry.
            raise ContractViolation(f"no provenance for {d.correlation_id}")
        out.append(AttributedDecision(decision=d, provenance=entry))
    return out
```

The failure this prevents is the opposite of a leak and just as expensive: a blinding scheme that *destroys* the provenance — hashing the provider into oblivion, or stripping it before the record was ever durably written — leaves an audit that cannot answer "which vendor's candidates were accepted?" Blinding removes the field from *a view*; the system of record keeps it, under the same versioning and identity discipline as everything else (`canonical-identity.md`).

Ordering matters and is worth stating explicitly in the design: the ledger is written **before** the blinded stage runs, and the rejoin happens **after** it closes. A rejoin available mid-stage is a lookup any component can perform, which un-blinds by API rather than by field.

### 6. Blinding and authority are the same construction

"This consumer must not *see* X" and "this producer must not *say* Y" are enforced identically: **absent from the schema, plus rejected on arrival.**

| Direction | Statement | Construction |
|---|---|---|
| Blinding (consumer-side) | The reviewer must not see `provider` | View type has no `provider`; allowlist projection; canary asserts the value never appears |
| Authority (producer-side) | The submitter must not state a `preferred_outcome` | Record type has no such field; parser rejects the key; authority test proves rejection |

Recognizing them as one construction is what stops a team from solving the first carefully and leaving the second to a comment. Both are also the same closure problem: a free-text field defeats either, an unshuffled batch order defeats either, a debug endpoint that returns the raw record defeats either.

## Rationalizations

| Rationalization | Reality |
|---|---|
| "We redact it at serialization" | The value is present in the object every consumer holds. One serializer redacts; the log line, the exception string, the debug endpoint, and the next serializer do not. Redaction is a filter on one path out of an unbounded set. |
| "The projection deletes the sensitive keys" | Deny-lists fail open. Every field added after the deny-list was written is exposed by default, by a diff that touched neither the projection nor the reviewer. Allowlist, field by field. |
| "The field is deprecated, nobody reads it" | Deprecated means "still on the wire". "Nobody reads it" is a statement about today's code, made on behalf of every future edit and every future contributor. Delete it from the view type. |
| "It's just a notes field for humans" | It is an unlimited, unversioned, untyped channel that carries exactly what your typed fields exclude — and it looks innocuous in review because free text is *supposed* to be arbitrary. Closed enums, or prose outside the boundary. |
| "We ignore that field if the producer sends it" | Then a producer keeps sending it forever and nobody learns they tried. Rejecting turns an unauthorized influence attempt into a named, attributable violation. |
| "Blinding is easy — we just hash the provider before writing" | That is not blinding, it is destroying provenance. The blinded stage cannot see it; the auditor afterwards cannot either. Hold provenance outside the view, keyed by correlation ID, and rejoin after. |
| "The type checker will catch anyone touching `.provider`" | Only on the paths that are typed and checked. `asdict()`, `**kwargs`, JSON round-trips, and template rendering all evade it. Absence from the view type is the enforcement; the type checker is a convenience on top. |

## Red flags

- `dataclasses.asdict()`, `model_dump()`, `vars()`, or `__dict__` anywhere on the producing side of a blinded boundary.
- A projection built as copy-then-`del`, copy-then-`pop`, or a `EXCLUDED_FIELDS` / `DENY_LIST` constant.
- `= "<redacted>"`, `"***"`, or a `redact()` helper applied at serialization time.
- A `notes` / `detail` / `context` / `rationale` string on a record that also has carefully typed, carefully blinded fields — the tell is the contrast.
- A comment of the form `# do not read this downstream`, or a field annotated deprecated but still emitted.
- A blinded-boundary test that asserts only absence, with no positive control proving the sentinel was ever in the source.
- Correlation IDs derived from, or containing, the blinded value; batches whose order reflects the blinded attribute.
- A blinding scheme with no answer to "how do we attribute this decision afterwards?"

## Quick reference

| Situation | Correct construction |
|---|---|
| Consumer must not see a field | Separate view type with no such field; hand-written allowlist projection as the view's only constructor |
| Nested records reachable from the source | Field-policy registry + closure walk (`get_type_hints`, not `Field.type`); undeclared field fails at import and in CI |
| Proving the blinding holds | Sentinel-value canary asserted absent from serialized view, errors, IDs, and metric labels — plus a positive control |
| Producer must not express an influence | No such field in the schema **and** parser rejects the key; authority test per forbidden name per version |
| Human explanation needed | Closed-enum reasons + numeric-only score maps at the boundary; prose lives outside, keyed by correlation ID |
| Attribution needed after the blinded stage | Provenance ledger written before the stage, held outside the view, rejoined by correlation ID after it closes |
| Changing what a blinded view exposes | Schema version bump on the view; widening is a semantic change with a review, not a convenience |

## Cross-references

- `contract-testing.md` — authority tests (forbidden fields cannot arrive), canary tests with positive controls, and CI enforcement of the field-policy closure.
- `deterministic-resolution.md` — covert-channel closure for resolvers; why no free text or passthrough field may reach resolution, and why batch position and ordering leak.
- `silent-default-elimination.md` — the closed-enum rule applied to absence reasons; an absence reason with a free-text detail field is this sheet's §4 defect.
- `schema-versioning-and-evolution.md` — view schemas are versioned records; widening a blinded view is a version bump, and forbidden-key rejection is per-version.
- `canonical-identity.md` — opaque, content-independent correlation IDs; keeping the system of record's identity intact while the view is blinded.
- `versioned-policy-parameters.md` — what a consumer is permitted to see is policy; changing it is a recorded, versioned event.
