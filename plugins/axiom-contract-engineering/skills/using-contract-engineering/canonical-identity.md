---
name: canonical-identity
description: Use when the same semantic object arrives twice and the system treats it as two things — duplicate work, broken dedup, equivalence classes that don't hold — or when an edited record keeps its old identifier, or when a downstream stage cannot join its output back to the input that produced it, or when "which raw input did this come from" has no recorded answer. Covers semantic vs storage identity, canonicalisation before hashing, false-split and false-merge as testable properties, hash-function versioning, chain binding, and raw-to-canonical traceability.
---

# Canonical Identity

**The identity of a semantic object is a content-addressed hash of its canonicalised semantic content — and every downstream record derived from that object carries the same hash as an explicit field, so the whole chain joins on identity instead of on trust.**

## When this earns its cost

Read this sheet when:

- The same semantic object can be produced twice — by two producers, two runs, or a retry — and you need the system to know it is the same object.
- Dedup is "working" but the duplicate count keeps rising, or an equivalence class that should have three members has seven.
- An object was edited in place and kept its identifier, so a cached artifact, a QA verdict, or an approval now describes content that no longer exists.
- You are joining stages of a pipeline on a foreign key that names a row rather than a meaning.
- An audit asks "which raw input produced this deployment" and the answer requires someone's recollection.
- Two stages disagree about what a record is, and there is no mechanical way to detect that they disagree.

Running example throughout: a candidate pipeline — **raw proposal → canonical spec → compiled artifact → QA report → decision → deployment record**.

## The discipline

### 1. Semantic identity is not storage identity

Both exist. They have different jobs and neither substitutes for the other.

| | Storage identity | Semantic identity |
|---|---|---|
| Examples | `spec_id` UUID, autoincrement PK, S3 key, row version | `canonical_hash` over the semantic subset |
| Names | *this record instance* — a row, a write, an insertion event | *this meaning* — what the object says, regardless of who wrote it or when |
| Assigned by | The store, at write time | The content, deterministically, before the write |
| Two identical objects | Two IDs | One hash |
| One object, edited | Same ID | New hash |

Using a storage ID as semantic identity breaks in both directions at once:

- **Dedup and equivalence break.** Two producers submit the same proposal; the store mints `spec-a1` and `spec-b7`. Nothing in the system can see they are the same thing, so it compiles both, QAs both, and reports "two candidates evaluated" when one was evaluated twice. Every aggregate computed over that set is wrong, and it is wrong in the direction of looking productive.
- **Integrity breaks.** Someone edits `spec-a1` in place. The ID is unchanged, so the compiled artifact keyed to `spec-a1`, the QA report referencing `spec-a1`, and the approval granted to `spec-a1` all now point at content nobody approved. The record says the review happened; the review happened to something else.

Content addressing fixes both by construction: identity *is* content, so identical content is one identity and changed content is a different one. Editing an object does not mutate an identity — it produces a new object, which is exactly the truth you want recorded.

Keep the storage ID. You still need to name the physical record (for retrieval, for storage-layer audit, for "which write introduced this"). Just never join on it when the question is semantic.

### 2. Canonicalise before hashing — and declare the semantic subset

Hashing the bytes you happen to have serialized is not content addressing. It is addressing *this serialization of* the content, which changes when a library upgrades its float formatter. Two steps, in order:

**Step one: declare which fields carry meaning.** This is a contract decision, written down, reviewed, and versioned — not an implicit consequence of which fields the dataclass happens to have.

**Step two: canonically serialize exactly that subset**, and hash those bytes.

```python
import hashlib
import json
from typing import Any, Mapping

# --- Step one: the semantic subset, declared explicitly ---------------------

SEMANTIC_FIELDS: frozenset[str] = frozenset({
    "objective",          # what the spec is for
    "parameters",         # the tunable values that define behaviour
    "constraints",        # the bounds the artifact must satisfy
    "target_profile",     # the environment class it is built for
    "input_refs",         # canonical hashes of upstream semantic inputs
})

# Everything below is deliberately EXCLUDED, with the reason recorded.
# An exclusion without a reason is a false merge waiting to be discovered.
EXCLUDED_FIELDS: Mapping[str, str] = {
    "spec_id":            "storage identity — names the row, not the meaning",
    "created_at":         "when it was written; the same spec written twice is one spec",
    "producer_instance":  "which worker emitted it; identical output from two workers is one object",
    "producer_version":   "tool version is provenance, recorded on the record, not identity",
    "compile_strategy":   "how the artifact was built, not what was specified",
    "token_spend":        "accounting for the production run, not a property of the spec",
    "raw_ref":            "traceability link (see §6); does not change what the spec says",
}

CANONICAL_HASH_VERSION = "v3"      # see §4 — the function itself is versioned
CANONICAL_HASH_BITS = 256          # declared width — never truncated below this


def _canonicalise(value: Any) -> Any:
    """Normalise a value into its canonical semantic form.

    Every rule here is a contract decision. Changing any of them changes
    what identity means and is a hash-function version bump.
    """
    if isinstance(value, Mapping):
        # Key order is not meaning: sort. Applied recursively.
        return {k: _canonicalise(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        # ORDERED collection: order IS meaning. Preserve it.
        return [_canonicalise(v) for v in value]
    if isinstance(value, (set, frozenset)):
        # UNORDERED collection: order is NOT meaning. Impose a total order
        # on the canonical form so two equal sets serialize identically.
        return sorted(
            (_canonicalise(v) for v in value),
            key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, bool):
        return value                          # bool before int — bool IS an int
    if isinstance(value, (int, float)):
        # ONE numeric branch, not two. `1` and `1.0` describe the same value;
        # routing them through different branches is a false split, and it is
        # the most common one in JSON pipelines (encoders disagree on which
        # they emit). Declared decisions in this rule:
        #   - int and float unify: identity is the numeric value, not its type.
        #   - `+ 0.0` collapses -0.0 to 0.0; the sign of zero is not meaning.
        #   - NaN is REJECTED, not encoded: repr() maps every NaN to "nan",
        #     which would merge distinct records, and NaN != NaN makes it
        #     unusable as an identity component either way.
        #   - repr-shortest round-trips exactly and is stable across platforms.
        #     Fixed precision is the other legitimate choice — pick ONE, and
        #     changing the choice later is a hash-function version bump.
        # Caveat this domain accepts explicitly: unifying via float merges
        # integers beyond 2**53. A domain with large exact integers must
        # branch them separately — and declare that as its rule.
        f = float(value) + 0.0
        if f != f:
            raise ContractViolation("NaN cannot participate in semantic identity")
        return repr(f)
    if isinstance(value, str):
        return ALIASES.get(value, value)      # aliases resolve BEFORE hashing
    return value


ALIASES: Mapping[str, str] = {"prof-a": "profile_alpha", "use1": "us-east-1"}


def canonical_hash(record: Mapping[str, Any]) -> str:
    """Content-addressed semantic identity of `record`.

    Missing semantic fields are a contract violation, not an empty default —
    silently omitting a field from the hash input is a false merge.
    """
    missing = SEMANTIC_FIELDS - record.keys()
    if missing:
        raise ContractViolation(f"cannot hash: semantic fields missing {sorted(missing)}")

    subset = {k: _canonicalise(record[k]) for k in sorted(SEMANTIC_FIELDS)}
    canonical_bytes = json.dumps(
        subset,
        sort_keys=True,
        separators=(",", ":"),        # no insignificant whitespace
        ensure_ascii=False,           # one text encoding decision...
    ).encode("utf-8")                 # ...and it is UTF-8, always
    digest = hashlib.blake2b(canonical_bytes, digest_size=CANONICAL_HASH_BITS // 8)
    return f"{CANONICAL_HASH_VERSION}:{digest.hexdigest()}"   # version travels WITH the hash
```

Three properties this buys, each load-bearing:

- **Wire dialect does not leak into identity.** A v2 payload and a v3 payload describing the same spec must produce the *same* canonical hash, because each version's parser converts to the canonical internal representation first and identity is computed from that (`schema-versioning-and-evolution.md`). What gets hashed is never the wire form.
- **Identity is a total order.** Any place that needs a deterministic tie-break can sort by canonical hash rather than by arrival order or iteration order — a content-derived ordering, not a covert channel (`deterministic-resolution.md`).
- **Collision posture is stated, not assumed.** 256 bits is declared in the contract. Truncate it — to 64 bits for a shorter key, to 8 hex characters "for readability in logs" — and collisions move from theoretical to operational. Hash equality is proof of full semantic equality only under a declared width; a system that truncates and still treats equality as proof has silently swapped a guarantee for a hope.

### 3. Two failure directions, both testable properties

Canonicalisation can be too weak or too strong, and the two failures look nothing alike.

**Under-canonicalisation → false split.** Semantically equivalent objects hash differently. The equivalence class shatters; dedup stops deduplicating; cached artifacts miss; the same work runs N times and the system reports N distinct candidates. Causes, in rough order of frequency: field order surviving into the bytes, a non-semantic field (timestamp, producer instance, storage ID) leaking into the subset, float formatting varying across platforms or library versions, an alias not resolved before hashing, an unordered collection serialized in insertion order.

**Over-canonicalisation → false merge.** Semantically different objects hash the same. Two genuinely distinct specs collapse into one identity; the QA report for one is joined to the decision about the other; the difference between them becomes invisible to every downstream query. Causes: a field that *does* carry meaning excluded from `SEMANTIC_FIELDS`, or normalisation too aggressive to preserve a distinction that matters — lowercasing an identifier that is case-significant, sorting a list whose order is semantic, rounding a float below the precision the domain actually distinguishes.

False merge is the more dangerous of the two because it is *quiet*. A false split produces visible symptoms — duplicate counts, cache-miss rates, redundant work. A false merge produces a smaller, cleaner-looking dataset in which two things are silently one, and nothing in the system is shaped to notice.

Both are properties, and both belong in the test suite rather than in review judgement (`contract-testing.md`):

```python
def test_equivalence_invariance(spec):
    """Under-canonicalisation guard: transformations that preserve meaning
    must preserve identity."""
    for variant in (
        permute_mapping_order(spec),          # key order is not meaning
        substitute_aliases(spec),             # "prof-a" == "profile_alpha"
        reserialize_via_other_encoder(spec),  # float/format variance
        retype_integral_floats(spec),         # 1 and 1.0 are one value
        with_new_storage_id(spec),            # instance != meaning
        with_later_timestamp(spec),
        with_different_producer_instance(spec),
    ):
        assert canonical_hash(variant) == canonical_hash(spec)


def test_separation(spec):
    """Over-canonicalisation guard: every semantic field must be able to
    change the hash. A field nobody can perturb into a new identity is a
    field that is not participating in identity."""
    for field in SEMANTIC_FIELDS:
        assert canonical_hash(perturb(spec, field)) != canonical_hash(spec)
```

**Expect the false-split fix to iterate, and plan for that.** The first canonicalisation you write will have residual splits. They are found by property tests over realistic corpora — not by reading the function, because the leak is always the case you did not picture (the one platform whose float repr differs, the alias introduced by a migration two quarters ago, the collection you assumed was ordered). Each fix changes what the function computes, so **each fix is a hash-function version bump**, not a patch. A canonicaliser reaching v3 or v4 through successive property-test findings is a normal trajectory and a sign the tests are working; a canonicaliser still at v1 after a year of production traffic more often means nobody is testing for splits than that it was right the first time.

### 4. The canonicalisation function is versioned

Identity depends on the function that computes it, so that function is a versioned artifact and every record carries which version produced its hash.

```python
@dataclass(frozen=True)
class CanonicalSpec:
    spec_id: str                    # storage identity
    canonical_hash: str             # "v3:9f2c…" — version is inside the value
    hash_fn_version: str            # "v3" — also an explicit field, for indexing
    schema_version: int             # the WIRE version this was parsed from
    ...
```

Two axes, orthogonal, and conflating them causes exactly the defects this sheet is about:

- **Schema version** names the wire dialect. Same meaning arriving as schema v2 or v3 → *identical* canonical hash, because both parse into canonical internal form before hashing.
- **Hash-function version** names the canonicalisation rules. The same object under `canonical_hash_v2` and `canonical_hash_v3` → *different* hashes, and **both are valid under their declared version**.

So changing canonicalisation is a versioned migration event, not a correction (`schema-versioning-and-evolution.md`): old hashes remain the correct identities of their records under `v2`; new records get `v3`; re-hashing the historical corpus under `v3` is a deliberate, recorded backfill that writes the new hash *alongside* the old rather than overwriting it. Comparing a `v2` hash to a `v3` hash is never valid — inequality across versions means nothing, and equality across versions would be an accident.

The corollary: **one hash registry per canonicalisation version.** A dedup index, a content-addressed artifact store, or an equivalence-class table that mixes `v2` and `v3` hashes in one keyspace silently reintroduces false splits for every record whose canonicalisation changed. Either the version is part of the key (as in the `"v3:…"` prefix above) or the registry is scoped to a single version.

### 5. Binding across the chain

Identity earns most of its value downstream. Every record derived from the canonical spec carries the **same** canonical hash as an explicit field:

```python
@dataclass(frozen=True)
class CompiledArtifact:
    artifact_id: str
    spec_canonical_hash: str        # stamped from the INPUT spec, not recomputed
    ...

@dataclass(frozen=True)
class QaReport:
    report_id: str
    spec_canonical_hash: str        # carried forward from the artifact
    artifact_id: str
    ...

@dataclass(frozen=True)
class Decision:
    decision_id: str
    spec_canonical_hash: str        # carried forward from the QA report
    outcome: Outcome
    ...

@dataclass(frozen=True)
class DeploymentRecord:
    deployment_id: str
    spec_canonical_hash: str        # carried forward from the decision
    decision_id: str
    ...
```

**Stamped at creation from the input record — not recomputed from scratch at each stage.** The distinction is precise, and both halves matter:

- *Recompute-as-identity-assignment is the defect.* If stage 3 independently canonicalises whatever it has in hand and calls the result the identity, then when stage 3's view has drifted from stage 1's — a field mutated in transit, a different hash-function version deployed to that service, a re-parse under a different schema branch — you get two coexisting "identities" for one object and no signal that anything is wrong. The chain still joins, on the wrong thing.
- *Recompute-and-assert-equal is good practice.* A stage may recompute the hash from its input and **assert it equals the stamped value**, failing loudly on mismatch. That is a drift detector. The rule is not "never hash again"; it is "the stamped value is the identity, and any recomputation that disagrees with it is an alarm, not an alternative answer."

Joining the chain is then an equijoin on `spec_canonical_hash`: every artifact, report, decision, and deployment for one meaning, retrieved mechanically, with the join itself proving that each stage was operating on the same object. An audit that *can* join on identity produces evidence. An audit that must instead follow a trail of per-stage foreign keys, each assigned by whichever service wrote the row, produces testimony — a narrative that the stages were talking about the same thing, unverifiable by anyone who wasn't there.

The bar to hold to: **if any two records in the chain describe the same semantic object, they share a field whose equality is checkable without trusting the process that wrote either one.**

### 6. Raw-to-canonical traceability

Canonicalisation is lossy by design — it discards everything outside the semantic subset. That makes the raw input irreplaceable, so the canonical record links back to it explicitly:

```python
@dataclass(frozen=True)
class CanonicalSpec:
    canonical_hash: str
    raw_ref: str                     # storage ID of the exact raw proposal
    raw_digest: str                  # hash of the raw BYTES — pins the exact input
    transform_version: str           # which raw→canonical transform ran
    transform_report_ref: str        # what it normalised, dropped, or rejected
    hash_fn_version: str
    schema_version: int
```

Rules that keep this real rather than decorative:

- **Never delete the raw record because the canonical one exists.** Every question of the form "was this normalisation right?", "did the producer actually say that?", or "what did we drop?" is answerable only from the raw input. The canonical record is a *view*; deleting the source to save storage converts every future integrity question into an unanswerable one.
- **Pin the raw bytes, not just the raw ID.** A storage ID can be re-pointed or the row edited; `raw_digest` makes the linkage verifiable rather than merely stated.
- **Record the transformation, not only its result.** The transform report is what makes a false merge diagnosable after the fact: if two raw proposals collapsed to one canonical hash, the reports say which normalisation step erased the difference.
- **The raw→canonical transform is itself versioned** and its version travels on the record, for the same reason the hash function's does — re-running an old raw input through today's transform is a different operation from what originally happened.

## Rationalizations

| Rationalization | Reality |
|---|---|
| "We already have a UUID, that's the identity" | It is the identity of a *row*. Two identical objects get two UUIDs and dedup never fires; one edited object keeps its UUID and every prior approval now describes content that no longer exists. |
| "We hash the JSON we send — that's content-addressed" | That is addressing one *serialization*. A library that changes float formatting, a map that iterates differently, or an alias left unresolved silently splits the equivalence class, and the split shows up as duplicate work nobody attributes to hashing. |
| "The timestamp in the hash is harmless, it's just metadata" | It guarantees every emission is a new identity. Dedup, caching, and equivalence classes are all disabled by exactly that one field, and the system looks busy instead of broken. |
| "Each stage recomputes the hash, so they always agree" | They agree only while nothing drifts — which is precisely when you did not need the check. When a field mutates in transit or a service runs an older hash-function version, recomputation manufactures a second identity instead of raising an alarm. |
| "Hashes never collide, so equality means equality" | Only at a declared width. Someone will truncate to 8 hex characters for a log line and then join on it. State the width in the contract and treat equality as proof only under that width. |
| "We fixed the canonicaliser, no need to bump the version" | The fix changed what identity *means*. Unbumped, one version name now covers two identity functions and no consumer can tell which produced a stored hash — the version-in-name-only defect, applied to identity. |
| "We can drop the raw proposals once the canonical specs exist" | Canonicalisation is lossy. Every question about whether the normalisation was correct becomes unanswerable the moment the raw record is gone, and those questions arrive precisely when something has gone wrong. |
| "One hash index is simpler than one per version" | Until it holds `v2` and `v3` hashes of the same objects, at which point it reports every re-canonicalised record as new work. Version-prefix the key or scope the registry. |

## Red flags

- A join across pipeline stages keyed on a per-stage row ID, or a foreign key chain that must be walked in order to relate a deployment back to a spec.
- `hash(json.dumps(record))` with no `sort_keys`, no float rule, no field allow-list — or a hash over the record's `__dict__`.
- Any timestamp, storage ID, producer instance, spend/accounting field, or build-strategy field reachable from the hash input.
- An identity function with no version, or a version constant unchanged across a diff that alters normalisation, the field set, or the float rule.
- Downstream records that recompute identity instead of carrying it, or that recompute and *discard* a mismatch rather than raising.
- Records mutated in place while retaining their identifier.
- A dedup index, artifact store, or equivalence table whose keys do not encode a canonicalisation version.
- A canonical record with no link to the raw input, or a retention policy that deletes raw inputs once canonicalised.
- No property test that perturbs a semantic field and asserts the hash changes — the field set has never been proven to be doing anything.

## Quick reference

| Question | Answer |
|---|---|
| What names a row? | Storage identity (UUID, PK). Never used for semantic joins. |
| What names a meaning? | `canonical_hash` over the declared semantic subset. |
| What participates in identity? | An explicit `SEMANTIC_FIELDS` declaration; every exclusion carries a written reason. |
| Excluded by default | Timestamps, storage IDs, producer instance/version, spend/accounting, compilation strategy, traceability links. |
| Serialization rules | Sorted keys (recursive), one committed float encoding, aliases resolved pre-hash, unordered collections totally ordered, no insignificant whitespace, UTF-8. |
| Equivalent objects hash differently | False split (under-canonicalisation) — caught by equivalence-invariance property tests. |
| Different objects hash the same | False merge (over-canonicalisation) — caught by separation property tests. |
| Canonicalisation changed | Hash-function version bump; old hashes stay valid under their version; registries keyed or scoped by version. |
| Wire schema changed, meaning didn't | Same canonical hash — parse to canonical internal form first, then hash. |
| Downstream records | Carry the input's canonical hash, stamped at creation; recompute only to assert-equal and alarm on mismatch. |
| Where did this come from? | `raw_ref` + `raw_digest` + transform version + transform report; raw records are never deleted. |

## Cross-references

- `schema-versioning-and-evolution.md` — hashing the canonical internal representation rather than the wire dialect; canonicalisation changes as versioned migration events; the version-in-name-only defect this sheet inherits.
- `deterministic-resolution.md` — canonical serialization of resolver inputs and outputs; canonical identity as the deterministic tie-break ordering.
- `contract-testing.md` — equivalence-invariance and separation property tests, corpus-based split hunting, golden-hash fixtures.
- `silent-default-elimination.md` — why a missing semantic field is a violation at hash time, never an empty default.
- `versioned-policy-parameters.md` — binding the policy version in force onto decision records, alongside the canonical hash.
- `definition-lifecycle.md` — moving a change to the semantic subset from draft to locked before it affects identity.
- Cross-pack: `axiom-determinism-and-replay` — canonical encoding as a determinism channel; `axiom-audit-pipelines` — content-addressed records as audit evidence.
