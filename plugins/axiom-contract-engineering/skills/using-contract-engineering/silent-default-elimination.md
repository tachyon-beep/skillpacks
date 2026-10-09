---
name: silent-default-elimination
description: "Use when designing or reviewing any cross-boundary record where a field can be unmeasured, unavailable, or omitted \u2014 metrics, telemetry, sensor readings, optional measurements \u2014 or when a consumer crashes on a missing field and the tempting fix is a default value or a lenient .get(). Covers explicit absence encoding, validity masks, version-gated absence semantics, and the prohibition on tolerant readers."
---

# Silent-Default Elimination

**A default value at a boundary is a lie the producer never told. "Unmeasured" and "zero" are different facts; any encoding that cannot distinguish them will eventually act on the wrong one — and the action will look correct in every log.**

## When this earns its cost

Read this sheet when:

- A record crossing a subsystem boundary has fields that are sometimes unmeasured: a latency that can't exist when no requests occurred, a probe that samples every Nth tick, a sensor that a given host doesn't have.
- A consumer is crashing on a missing field and the on-call fix on the table is `payload.get(field, 0)` or a schema default.
- You are reviewing a parser described as "lenient", "tolerant", "forgiving", or "robust to older producers".
- A downstream decision system — autoscaler, trainer, adjudicator, alerting — consumes measurements and its wrong-but-plausible output would be expensive.

This failure class has killed real systems. The canonical death: a pipeline where plumbing quietly taught downstream consumers that *unmeasured means zero*. Every individual reader looked reasonable. The aggregate system made confident decisions on data that was never measured, and nothing failed loudly enough to notice until the decisions had compounded.

## The defect class

A **silent default** is any mechanism that converts *absence of information* into a *plausible value* without recording that the conversion happened:

```python
# Every one of these is the same defect:
cpu = payload.get("cpu_util", 0.0)            # reader-side default
@dataclass
class Metrics:
    cpu_util: float = 0.0                     # schema-side default
latency = row["latency"] or 0.0               # null-coalescing to a value
proto: `optional double cpu = 3;`             # proto3 scalar: absent reads as 0.0
score = compute(m) if m.valid else 0.0        # computed default
```

The defect is not the specific value. `-1` sentinels, `NaN`-that-gets-averaged, empty-string, and "last known value" without an age bound are all the same class: the consumer receives something value-shaped where no value exists, and every downstream computation proceeds as if it were measured.

Why it survives review: each individual default is locally defensible ("the old collectors don't send this field yet", "zero is a safe fallback"). The system-level consequence — decisions actuated on fabricated data — is visible in no single diff.

## The discipline

### 1. Absence is a value of a different type, never a value of the same type

Encode "unmeasured" so the type system forces every consumer through a branch that acknowledges it:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Union

class AbsenceReason(str, Enum):
    """Closed set. No free-text detail field — free text at a boundary is a
    covert channel (see blinding-by-construction.md)."""
    NOT_APPLICABLE = "not_applicable"   # no requests occurred; latency undefined
    NOT_SAMPLED = "not_sampled"         # probe cadence; this tick wasn't sampled
    PROBE_FAILED = "probe_failed"       # measurement attempted, errored
    UNSUPPORTED = "unsupported"         # this producer cannot measure this

@dataclass(frozen=True)
class Measured:
    value: float
    measured_at: float

@dataclass(frozen=True)
class Absent:
    reason: AbsenceReason

Maybe = Union[Measured, Absent]
```

A consumer cannot write `m.cpu_util * 2` — there is no `.value` on `Absent`. The compiler/type-checker turns "forgot to handle absence" from a production incident into a red squiggle.

`Measured(0.0)` and `Absent(...)` are now different facts, which is the entire point: **zero is a real measurement**, and the discipline must preserve its meaning too. A scheme where `0.0` is "suspiciously maybe-missing" is as broken as one where absence reads as zero.

### 2. Validity masks for dense/batch data

Tagged unions don't fit tensors and column batches. There, the contract carries an explicit **validity mask** alongside the values, and the two travel together or not at all:

```python
@dataclass(frozen=True)
class MetricBatch:
    values: "ndarray"        # shape [N, F]
    validity: "ndarray"      # shape [N, F], bool — True iff values[i,j] was measured
    schema_version: str

def masked_mean(batch: MetricBatch, col: int) -> Measured | Absent:
    mask = batch.validity[:, col]
    if not mask.any():
        return Absent(AbsenceReason.NOT_SAMPLED)
    return Measured(value=float(batch.values[mask, col].mean()), measured_at=...)
```

Rules that make masks real rather than decorative:

- **Every aggregation consumes the mask.** A mean over the raw column is a silent default with extra steps — the unmeasured cells hold *some* number (0? stale? uninitialized?) and it just averaged them in.
- **Masked-out cells hold poison, not zero.** Fill them with `NaN` (float) or a trap value in debug builds, so any code path that ignores the mask fails loudly in testing instead of computing plausibly in production.
- **The mask is not optional.** A `validity: ndarray | None = None` field where `None` means "all valid" reintroduces the defect one level up — omission of the mask silently asserts universal validity.

### 3. Fail loud at the boundary, not deep in the consumer

The parser is the last point where "this field is missing" is still a *fact about the message*. Past the parser it becomes a fact about your process state, unattributable to any producer. So:

- A field the schema declares mandatory and the payload omits → **typed parse error naming the field**, not a default, not `KeyError` three stack frames later.
- A field the schema declares absent-able → parsed to explicit `Absent`/mask-false.
- There is no third category. "Optional, defaults to X" at a boundary means X will eventually be actuated as if measured.

```python
class ContractViolation(Exception):
    """Raised at the boundary. Carries field name, record identity, producer,
    and schema version — everything needed to attribute the violation."""

def parse(raw: Mapping) -> MetricsRecord:
    try:
        host_id = str(raw["host_id"])          # mandatory: absence is an error
    except KeyError:
        raise ContractViolation("host_id", record=raw) from None
    cpu = _tagged_metric(raw, "cpu_util")      # absent-able: absence is Absent(...)
    ...
```

### 4. Absence semantics are version-gated

This is the rule strong engineers miss even when they get everything above right. **A reader may treat absence as meaningful only under a schema version that declares that absence.**

The failure shape: a new producer version legitimately stops sending `gpu_util` (those hosts have no GPU). The consumer's fix is `payload.get("gpu_util")` → `None` → handled. Looks disciplined — absence is explicit, nobody defaulted to zero. But the reader now accepts absence from *every* payload, including a v1 producer that lost the field through a serialization bug. Legitimate schema evolution and silent data loss have become indistinguishable. Absence got its meaning from context inference ("those hosts have no GPU") instead of from the contract.

```python
# WRONG — absence tolerated unconditionally:
gpu = payload.get("gpu_util")    # any producer may now silently drop the field

# RIGHT — absence meaningful only where the schema version declares it,
# and versions are ENUMERATED, never ranged (`>=` would parse a future v4
# under v3 semantics — the fail-open gate schema-versioning-and-evolution.md
# forbids; a new version joins this branch only after its changelog is read):
if version == 3:                 # v3 declared gpu_util absent-able
    gpu = _tagged_metric(payload, "gpu_util")   # explicit Absent(...) allowed
elif version == 2:               # v2 declares gpu_util mandatory
    gpu = _require_metric(payload, "gpu_util")  # absence here is a violation
else:
    raise ContractViolation(f"unsupported version {version}")  # v4 lands here
```

Under incident pressure this costs one extra branch over the `.get()` one-liner. That branch is the entire difference between "the fleet migrated" and "we stopped noticing when producers drop fields."

### 5. No tolerant readers

A **tolerant reader** fills gaps between what arrived and what the schema says, so that mismatched producer and consumer keep interoperating. Common forms: defaulting missing fields, accepting either of two field names (`payload.get("latency", payload.get("latency_ms", 0.0))`), coercing types quietly, treating an unknown or missing schema version as some assumed version.

Every one converts *contract drift* — the precise signal that producer and consumer no longer agree — into continued silent operation on reinterpreted data. The drift doesn't go away; it goes invisible, and it compounds until the reinterpretations disagree expensively.

The disciplined alternative is not brittleness for its own sake; it is that **compatibility decisions are made by the version gate, explicitly, once** — not by each field's reader, implicitly, forever. A reader handles exactly the versions it declares (see `schema-versioning-and-evolution.md` for the gate; unknown versions fail closed). Within a declared version, unknown *extra* fields may be preserved-and-ignored only if the version's evolution rules say additive fields are non-breaking — that is a documented contract decision, not reader charity.

## Consumer-side discipline: absence must change the decision

Encoding absence correctly is half the job. The consumer must then do something principled with it, and the principle is: **absence weakens confidence, and irreversible or expensive actions require confidence.**

- An adjudicator missing a required input **abstains** (explicit `INSUFFICIENT_DATA` outcome — itself a recorded, first-class result), it does not decide from the fields that happen to be present as if they were the whole picture.
- Asymmetric costs get asymmetric evidence requirements: if scale-*down* (or reject, or delete) is the expensive-when-wrong direction, absence blocks it rather than enabling it.
- If a stale value may be carried forward, the carry-forward is bounded by a declared age budget and the consumed value is **recorded as stale with its age** on the resulting decision — never presented as fresh. (And the carry-forward state must be recorded, not hidden in process memory — see `deterministic-resolution.md`.)
- Substituting a model-estimated or renormalized value for a missing input is a *policy change*: it alters what the output means. It requires a policy version bump (`versioned-policy-parameters.md`), not a quiet formula edit in a hotfix.

## Rationalizations — heard in real incident channels

| Rationalization | Reality |
|---|---|
| "Zero is a safe default here" | Zero is a *measurement claim*. The autoscaler that reads absent-cpu as 0.0 scales down the host it knows nothing about. Safe defaults at boundaries don't exist; there are only defaults whose damage you haven't traced yet. |
| "The old collectors just don't send it yet" | Then the version gate says so, for those versions, explicitly. A reader-side default "handles" old producers by also handling — forever, invisibly — every future producer that drops the field by accident. |
| "We'll add proper absence handling after the deadline" | The default ships, downstream consumers calibrate against fabricated values, and removing the default later *changes observed behavior* — which now needs its own migration. The cheap moment is now. |
| "`.get(field)` returning None IS explicit absence" | Only if the schema version declares that field absent-able. Unconditional `.get()` accepts absence from every producer version including broken ones — see §4. |
| "Fail-loud will page us constantly" | Fail-loud pages you exactly when producer and consumer disagree about the contract — which is precisely the event you need to know about. If that's frequent, the contract is wrong or the fleet is drifting; both are findings, not noise. |
| "The mask is on the record, consumers can check it" | *Can* is not a contract. If any aggregation path compiles without consuming the mask, one of them will skip it. Poison the masked cells and make the type system or the test suite prove every path checks. |

## Red flags — stop and redesign

- Any schema default on a field describing a measurement, in any IDL (`= 0.0` in a dataclass, proto3 scalar fields whose absence is unobservable, Avro field defaults used at read time).
- The words "lenient", "tolerant", "graceful fallback", or "backwards compatible" in a *parser's* description or comments.
- `.get(key, default)` on a wire payload, or `x or 0.0`, or a bare `.get(key)` with no version gate around it.
- A validity mask that is optional, or that any aggregation path does not consume.
- A test suite that round-trips a record built from the schema's own constructor (it can never exercise the absence paths — build fixtures from realistic wire payloads: old producer, new producer, partial payload).
- An abstain/insufficient-data outcome that is possible in principle but appears zero times in the decision log.

## Quick reference

| Situation | Correct encoding |
|---|---|
| Scalar field, sometimes unmeasurable | Tagged union `Measured \| Absent(reason)`; reason is a closed enum |
| Dense batch / tensor | Values + mandatory validity mask; poison masked cells; every aggregation consumes the mask |
| Mandatory field missing from payload | Typed `ContractViolation` at the parser, naming field + producer + version |
| Field legitimately absent in newer schema | Absence declared in schema version N; reader accepts absence only under the enumerated versions that declare it (never a `>=` range); violation under versions that declare it mandatory; unknown versions fail closed |
| Consumer missing a required input | Recorded abstain (`INSUFFICIENT_DATA`), never a decision from partial inputs presented as complete |
| Stale value carried forward | Declared age budget; consumed value recorded as stale-with-age; carry-forward state recorded |
| Imputed/renormalized substitute | Policy version bump; the substitution recorded on the output |

## Cross-references

- `schema-versioning-and-evolution.md` — the version gate this sheet's §4 relies on; unknown versions fail closed.
- `contract-testing.md` — absence-path fixtures, schema-invalid rejection tests, and why constructor-round-trip tests are vacuous.
- `deterministic-resolution.md` — recording resolution inputs (including staleness) so decisions are reproducible.
- `versioned-policy-parameters.md` — why imputation and renormalization are policy changes.
- `blinding-by-construction.md` — the closed-enum rule for absence reasons (free text is a covert channel).
