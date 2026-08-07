---
name: schema-versioning-and-evolution
description: Use when changing any cross-boundary record — adding, removing, or renaming fields, changing units, meaning, normalisation, basis, or provenance — or when writing the reader-side version gate, planning a mixed-fleet migration, or tempted by a compatibility shim, dual-format fallback, or "old readers will just ignore it". Covers what counts as a version bump, fail-closed gates, and expand/contract migration without legacy code paths.
---

# Schema Versioning and Evolution

**Any change in a field's meaning, width, unit, basis, normalisation, or provenance is a new schema version — and a reader confronted with a version it does not explicitly support fails closed. Every softer rule eventually lets two meanings share one name.**

## When this earns its cost

Read this sheet when:

- You are about to change a contract record and are deciding whether it "counts" as a version bump.
- You are writing the reader-side dispatch on `schema_version` and deciding what happens for versions you don't recognise.
- A fleet upgrades slowly and old producers will emit the previous format for months.
- Someone proposes a compatibility shim, a dual-format fallback (`payload.get("latency", payload.get("latency_ms"))`), or "we'll keep accepting both for a while".
- A field's unit, scale, or reference basis is changing (ms→s, fraction→percent, raw→normalised) — the highest-casualty change class, because the type doesn't change and nothing crashes.

## What is a version

A schema version names a **meaning**, not a shape. Two payloads with identical field names and types but different units are different schemas. Consequently:

**Bump the version for any of:**

- A field's **meaning** changes ("time to first byte" → "time to last byte").
- A field's **unit or scale** changes (ms → s; 0–100 → 0.0–1.0).
- A field's **basis** changes (per-request → per-tick aggregate; gross → net).
- A field's **normalisation** changes (raw → z-scored; and *which* normalisation manifest applies).
- A field's **provenance** changes (measured → imputed; direct → derived from other fields).
- A field's **absence semantics** change (mandatory → absent-able, or the reverse) — see `silent-default-elimination.md` §4.
- A mandatory field is added or removed.

**Additive-only changes** (a new absent-able field old readers may ignore) can be declared non-breaking *within* a major version — but that is a written rule of the contract ("minor additions: readers preserve-and-ignore unknown fields"), decided once, not reader charity improvised per field.

The version lives **in the record** (`schema_version` as an explicit field), and it is never defaulted at parse time: a payload with no version is a violation, not "presumably v1". Defaulting the version is the same silent-default defect applied to the contract itself.

### Version-in-name-only

The inverse defect: the schema changed and the version didn't. `SCHEMA_VERSION = "2.0"` bumped in March for a new field, then the latency unit changed in June with no bump — now `"2.0"` names two mutually incompatible meanings and **no consumer can disambiguate, ever**. There is no repair for a version that meant two things; you can only bump now and treat the ambiguous version as its worse interpretation. Prevention is a review rule: *a diff that touches a contract record's semantics and does not touch its version constant is rejected mechanically*, plus a golden-fixture hash test that fails when serialized meaning changes under an unchanged version (`contract-testing.md`).

## The reader-side gate: fail closed

The reader dispatches on the version and handles **exactly the versions it declares**:

```python
SUPPORTED = {2, 3}

def parse(raw: Mapping) -> MetricsRecord:
    try:
        version = int(raw["schema_version"])       # never defaulted
    except KeyError:
        raise ContractViolation("schema_version missing — payload rejected") from None
    if version == 3:
        return _parse_v3(raw)
    if version == 2:
        return _parse_v2(raw)
    raise ContractViolation(
        f"schema_version {version} not supported (supported: {sorted(SUPPORTED)})"
    )
```

The one that gets smart engineers is the **fail-open range gate**:

```python
# WRONG — and it looks like foresight:
if version >= 2:
    return _parse_v2(raw)   # "anything genuinely new lands in unknown_fields;
                            #  a breaking change will bump to 3 and get its own branch"
```

The comment is a promise made on behalf of *future producers* that the reader cannot enforce. When v3 ships with a changed unit, every un-upgraded reader parses it under v2 semantics — silently, with a test suite that passes, because someone even wrote `test_future_version_still_parses`. The whole value of the version field is that the *producer* declares meaning and the *reader* refuses meanings it doesn't know. `>=` forfeits that in one character. A reader that fails closed on v3 forces exactly the right event: the deploy halts, loudly, until the reader supporting v3 ships.

Fail-closed applies to every compatibility axis, not just the record version: grammar/profile versions, policy versions, device or format capabilities — an incompatibility between declared versions is a refusal, never a best-effort interpretation.

## No legacy shims: the consumer moves or the version gates

A **compatibility shim** is reader code that accepts two meanings for one name and reconciles them inline — the dual-key fallback, the try-both-units heuristic, the "if it looks like ms, divide by 1000" guess. Shims are how two meanings *coexist indefinitely*: nothing ever forces the old producer to finish migrating, the shim outlives everyone who remembers why it exists, and each shim multiplies against the others (three shimmed fields = eight possible payload dialects, none tested).

The discipline permits exactly two states for any consumer:

1. **The version gates.** The reader has an explicit `_parse_v2` branch because v2 producers still exist. That branch parses v2 *completely and correctly under v2 semantics* (including unit conversion into the canonical internal form) — it is a full parser for a declared version, not a patch on the v3 path. It has a **retirement condition**: measured evidence (a parse-version counter, not an assumption) that no v2 producer remains, then the branch is deleted.
2. **The consumer moves.** The reader drops the old version entirely and producers must upgrade first.

What is *not* a permitted state: field-level fallbacks inside one parse path, accepting versionless payloads, or "temporary" leniency with no retirement condition. If you cannot name the metric that will tell you when the old branch can be deleted, you are not gating a version — you are growing a shim.

## Migration: expand / contract, with meaning changes as renames

For a mixed fleet, semantic changes ship as **new names**, never as reinterpreted old ones:

1. **Never change what an existing name means.** To change a unit, add `latency_seconds` and retire `latency_ms`. There is no deployment ordering under which "same name, new meaning" is correct for a mixed fleet — some reader somewhere always applies the wrong interpretation to some producer.
2. **Expand:** producers dual-write old and new names (a minor, additive bump). Readers prefer the new name when present and never convert twice.
3. **Contract:** once the parse-version counter shows the old producers are gone, producers stop emitting the old name (major bump), readers delete the old branch.
4. **Retired names are never reused.** A resurrected name with a new meaning defeats every archived record and every log query written against the old one.

Internally, keep **one canonical representation** regardless of wire version — every version branch converts at the boundary (`_parse_v2` converts ms→s into the canonical `latency_s`). The consumer's logic never grows a unit branch; only parsers know wire dialects.

Archived records are read the same way: a stored decision or measurement is interpreted under **its own recorded version**, by the version branch that matches it — not under whatever the schema means today. This is why every derived record must carry the versions of what it consumed (`versioned-policy-parameters.md` extends this to policy).

## Rationalizations

| Rationalization | Reality |
|---|---|
| "`version >= N` is future-proof" | It is future-*blind*: it promises today's semantics on behalf of producers that don't exist yet. Fail closed; add branches when versions actually ship. |
| "It's the same field, just in seconds now" | Same name + new meaning = two schemas sharing one identifier. Some reader will apply ms-math to s-values. New name, expand/contract. |
| "We only added a field, no bump needed" | Only if the contract's written evolution rules declare additive fields non-breaking within the major version — and the addition still gets a minor bump so fixtures and consumers can pin it. |
| "The shim is temporary, just for the migration" | A shim with no measured retirement condition is permanent. Gate the version, count parses by version, delete on evidence. |
| "Old payloads have no version field, so default to v1" | A defaulted version is a guessed meaning. Emit the version from every producer you control; for a truly versionless legacy fleet, gate on a fingerprint you can verify (distinct envelope key set), not on absence. |
| "Failing closed will break the pipeline during rollout" | It halts the *mismatched* pairing — which is broken already, just silently. Deployment order (readers learn vN before producers emit it) is the fix, and fail-closed is what enforces that order. |

## Red flags

- `>=` or `<=` in a version gate; a `default=` on the version field; any parse path reachable without a version check.
- One parse function serving two versions via per-field fallbacks (`raw.get(new_key, raw.get(old_key))`).
- A unit, basis, or normalisation change in a diff that doesn't touch the version constant.
- A version branch with no retirement metric, or a migration plan with the word "eventually".
- A test asserting that an *unknown future* version parses successfully.
- Consumers reading archived records under current-version assumptions.

## Quick reference

| Change | Version action | Migration shape |
|---|---|---|
| New absent-able field | Minor bump (if contract declares additive-safe) | Ship; old readers preserve-and-ignore per contract rule |
| Remove a field | Major bump | Deprecate → measure zero emitters → delete |
| Unit / scale / basis / meaning change | New field name + major bump | Expand (dual-write) → measure → contract |
| Absence semantics change | Major bump | Reader gains version-gated absence branch |
| Normalisation / provenance change | Major bump (+ manifest reference in-record) | As meaning change |
| Unknown version at reader | — | `ContractViolation`; deploy halts; reader ships first |

## Cross-references

- `silent-default-elimination.md` — version-gated absence semantics; why the version field itself is never defaulted.
- `contract-testing.md` — golden-fixture hash tests that catch version-in-name-only; per-version fixture suites; parse-version counters.
- `versioned-policy-parameters.md` — the same discipline applied to policy and decision records.
- `definition-lifecycle.md` — how a new schema version moves draft → approved → locked before anything emits it.
- `canonical-identity.md` — why the canonical internal representation, not the wire dialect, is what gets hashed.
