# Canonical Encoding for Fingerprinting

## Overview

**A fingerprint is only as good as the bytes it covers. The bytes must be the same on every producer, every reader, every verifier, every export — forever.**

Canonicalisation is the rule that says: given a logical entry, there is exactly one byte sequence that represents it. Without that rule, two correct producers will compute different hashes for the same decision, and a verifier will reject both. With a sloppy rule, an attacker will mutate fields you forgot to canonicalise and your chain will validate the lie.

Hash function choice is downstream of canonicalisation. SHA-256 of the wrong bytes is no better than no hash at all.

**Default choice for new pipelines: RFC 8785 JSON Canonicalisation Scheme (JCS).** Deviate only with a documented reason (see *When not to use JCS* below).

## When to Use

Use this sheet when:

- Designing the byte form that goes into a fingerprint, signature, or chain link.
- A producer in language A and a verifier in language B must agree on hashes byte-for-byte.
- Exports cross trust boundaries and must verify standalone.
- You see "we'll just `json.dumps(...)`" in a design and need to explain why that fails.

Do not use this sheet for:

- Wire-format choice for non-audit traffic (use protobuf, msgpack, whatever — different problem).
- Storage format choice (the chain covers logical entries; storage may add framing).

## Core Principle

> The canonicalisation rule is part of the audit specification, not part of the implementation. It is versioned. Its byte output is the input to your hash function. Changing it is a chain-breaking event.

## Why `json.dumps()` Is Not Canonical

CPython's `json.dumps`, Node's `JSON.stringify`, Go's `encoding/json` all produce *valid* JSON that hashes differently:

- **Map ordering**: insertion order, alphabetical, language-default — varies by serialiser, by library version, by runtime.
- **Whitespace**: spaces between key/value, indentation, trailing newline.
- **Number form**: `1`, `1.0`, `1e0`, `1.000`, `+1` — JSON allows several; serialisers pick differently.
- **Unicode escapes**: `"é"` vs `"é"` vs `"é"` — same string, different bytes.
- **Locale-dependent floats**: `,` vs `.` decimal separators leak in non-`C` locale code paths.
- **Trailing zeros on floats**: `1.0` vs `1`, `1.10` vs `1.1`.

A hash over non-canonical JSON works *on the producer's machine* and breaks the moment a reader, verifier, or downstream consumer recomputes.

## RFC 8785: JSON Canonicalisation Scheme (JCS)

JCS is the mature, ratified canonical-JSON standard. It is the default unless you have a specific reason to deviate.

The full RFC is short and worth reading. The rules in summary:

1. **Object keys are sorted by their UTF-16 code-unit value, lexicographically.** Not Unicode codepoint, not language-default — UTF-16 code unit. This affects supplementary-plane characters; for ASCII keys it matches alphabetical sort.
2. **No insignificant whitespace.** No spaces, no indentation, no trailing newline.
3. **Strings: the JSON-string production with minimum-escape rule.** Most characters are emitted directly; only the JSON-mandatory escapes (`"`, `\`, control characters U+0000–U+001F) are escaped. Non-ASCII characters are emitted as UTF-8 bytes, not `\uXXXX` escapes.
4. **Numbers: the IEEE 754 double-precision form via the ECMAScript `Number.prototype.toString` algorithm.** This produces the shortest round-trippable form. `1.0` becomes `1`. `1.50` becomes `1.5`. `1e2` becomes `100`. NaN and ±∞ are not representable — see *Floats*.
5. **The output is UTF-8 with no BOM.**

The output is deterministic across every conformant implementation in any language. That is the property you are paying for.

### Conformant libraries

| Language | Library |
|----------|---------|
| Go | `github.com/cyberphone/json-canonicalization/go/src/webpki.org/jsoncanonicalizer` |
| JavaScript / TypeScript | `canonicalize` (npm) |
| Python | `rfc8785` (PyPI) — pure Python, mirrors the spec test vectors |
| Java | `org.webpki.jcs` |
| Rust | `serde_jcs` |

**Pin the library version in `02-canonical-encoding-spec.md`.** A library bug-fix that changes byte output is a chain-breaking event whether you noticed or not. Keep test vectors from the RFC's appendix in CI.

## Gotchas (Address Every One in `02-canonical-encoding-spec.md`)

### Floats

JCS represents numbers as IEEE 754 doubles, serialised via the ECMAScript shortest-round-trip rule. Three traps:

1. **NaN, ±Inf, -0** — JSON cannot represent these. JCS forbids them in the output. *Decide upstream*: either reject decisions that produce them at the schema layer, or normalise (e.g., `-0 → 0`, `NaN → null`) and document the rule. Do not let the serialiser choose.
2. **Precision beyond double** — values from `Decimal128`, `BigInt`, or financial fixed-point types lose precision when forced through double. *Encode such fields as strings* (e.g., `"amount": "12345.6789"`) so the canonical form preserves bytes. The decision: anything where rounding changes meaning belongs in a string.
3. **Floats from non-deterministic sources** — GPU reductions, parallel sums, JIT-recompiled code can produce slightly different doubles for the same logical computation. If the field is a fingerprint input, the producer must commit to a deterministic computation path or store the result with explicit rounding. See `partial-replay-from-trail.md` for replay implications.

### Map ordering

Already specified by JCS (UTF-16 sort). Two derived risks:

1. **Programmatic insertion-order assumptions** — code that builds an entry from `OrderedDict` and assumes the canonicaliser preserves it is wrong; JCS will reorder. Tests must construct entries in *random* key order and assert byte equality with a fixed reference.
2. **Nested objects** — sort applies recursively. A library that sorts only the top level produces non-canonical output for nested structures.

### Timezones

Timestamps are the most-mutated audit field, the most-litigated, and the most subtly broken.

1. **Always emit RFC 3339 UTC with explicit offset** — `2026-01-15T14:30:00.123Z`, never `2026-01-15T14:30:00`. Local time without offset is unverifiable. `Z` and `+00:00` are equivalent semantically but produce different bytes — *pick one in the spec and enforce it*.
2. **Sub-second precision is a schema decision, not a runtime decision** — `2026-01-15T14:30:00Z` and `2026-01-15T14:30:00.000Z` hash differently. Choose the precision once (typically milliseconds or microseconds), enforce in the schema, and reject entries that violate.
3. **Leap seconds and ambiguous times** — for high-frequency systems near midnight UTC, choose between Unix epoch (smears) and TAI; document the choice. Most pipelines use Unix-epoch milliseconds and accept the smear.
4. **Wall clock vs monotonic clock** — wall-clock time is the audit field; monotonic time is a separate field for ordering. Never substitute one for the other.

### Unicode

JCS pins UTF-8 with minimum escaping. Two further traps:

1. **Normalisation form** — `é` can be U+00E9 (precomposed) or U+0065 U+0301 (combining). They render identically and hash differently. *Decide on a normalisation form* (NFC is the standard choice for new pipelines) and apply it *before* canonicalisation. Document the form in `02-`. Without normalisation, copy-paste between editors silently breaks chains.
2. **Bidirectional and zero-width characters** — adversarial inputs may insert U+200E, U+200B, etc. Decide whether the schema forbids them, normalises them, or preserves them. The threat model in `07-` should address bidi-injection on string fields used in human review.

### Integer width

Different runtimes have different default integer widths. JCS forces all numbers through double, which is *correct for many fields and wrong for some*:

- **64-bit IDs, hashes, signatures** lose precision through double. Encode as hex strings or base-64.
- **Counters that may exceed 2^53** — encode as strings.
- **Booleans** — `true`/`false` only; `0`/`1` and `"yes"` are not booleans.

### Empty values

`null`, `""`, missing-key, and `[]` all hash differently. Schema must specify which is valid for each field. Common rules:

- Required fields are present and non-null.
- Optional absent fields are *omitted entirely*, not present-as-null.
- Empty collections are present-as-`[]` only when the schema explicitly says so; otherwise omit.

Picking one rule per field eliminates a class of replay attack where the adversary toggles between forms.

## When NOT to Use JCS

Cases where deviation is defensible:

- **Existing pipeline**: a system already emits canonical-form-X with extensive test vectors; switching is a chain-breaking migration. Stay on X. Document the deviation and pin its spec.
- **Non-JSON entries**: protobuf with deterministic-marshalling, CBOR (RFC 8949) with the deterministic encoding profile, or Avro with the canonical schema rule. All are legitimate alternatives. *None of them save you from the gotchas above* — float, timezone, integer-width, normalisation discipline applies regardless of envelope format.
- **Cross-organisation handoff**: the consumer mandates a specific form. Use theirs. Keep your test vectors.

A deviation must be recorded in `02-canonical-encoding-spec.md` with: the form chosen, the rationale, the conformance reference (spec or library + version), and the test-vector strategy. "We use Python's `json.dumps` with `sort_keys=True`" is **not** a canonical encoding; it is a near-canonical form with bugs (Unicode escape ambiguity, float-form variation). Do not record it as one.

## Test Vectors as the Discipline

Canonical encoding is the field where overconfidence is most expensive. Two protections:

1. **Keep the RFC 8785 appendix test vectors in CI** for whatever JCS implementation you use. If a library upgrade fails the vectors, *you have an incident*, not a flaky test. Pin the library version, regression-test before deploying.
2. **Keep your own test vectors of decision entries**: a frozen set of `(logical entry, expected canonical bytes, expected hash)` triples. These catch subtle regressions in your wrapping code (the schema-validation layer, the timestamp formatter, the field projector) that the library vectors do not. Add a vector each time you find a new gotcha.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `json.dumps(entry, sort_keys=True)` as canonical form | Use a real JCS library; keep RFC vectors in CI |
| Encoding floats from non-deterministic computation directly | Round explicitly at schema layer, document rounding rule |
| Allowing `2026-01-15T14:30:00.000Z` and `2026-01-15T14:30:00Z` interchangeably | Schema fixes one precision; reject the other |
| Unicode normalisation deferred to "the database does it" | Normalise (NFC) at entry construction, before canonicalisation |
| 64-bit IDs serialised as JSON numbers | Encode as hex strings; document in `02-` |
| Missing/null/empty treated as equivalent | Schema specifies one valid form per field |
| Library upgrade changes byte output without anyone noticing | Pin version; CI regression on RFC test vectors |
| Canonicalisation is "an implementation detail" — not in the spec | Lift it into `02-`; treat changes as chain-breaking events |

## Spec Output (`02-canonical-encoding-spec.md`)

The sheet's deliverable answers, in order:

1. **Form chosen** — RFC 8785 JCS (default) or named alternative with rationale.
2. **Library + version** — exact dependency name and pinned version, per language present in the system.
3. **Float discipline** — rounding rule, treatment of NaN/Inf/-0, fields encoded as strings.
4. **Timestamp form** — RFC 3339 UTC, explicit `Z` or `+00:00`, sub-second precision, wall-vs-monotonic separation.
5. **Unicode normalisation** — NFC (default) or named alternative; where the normaliser is invoked.
6. **Integer-width policy** — which fields are strings instead of numbers.
7. **Empty-value policy** — per field: required-present, optional-omitted, optional-null, optional-empty-array.
8. **Test-vector strategy** — RFC vectors + system vectors, where they live, when they run.
9. **Versioning rule** — how `02-` is versioned, what counts as a chain-breaking change, the migration approach for legacy entries (re-canonicalisation forbidden; legacy entries verify against the form they were written with — see `fingerprint-chains-and-integrity.md`).

Without these nine items the spec is incomplete and Check 3 of the consistency gate will fail.

## The Bottom Line

**Canonical bytes before hashes. RFC 8785 JCS by default, with documented gotchas covered, library versions pinned, test vectors in CI, and a versioning rule that treats encoding changes as chain-breaking events.**

---

**Retrieval test (run at end of build):** "Our pipeline currently uses `json.dumps(entry, sort_keys=True)` and SHA-256. We have entries spanning two years. Walk through the migration to RFC 8785 JCS — what's chain-breaking, what isn't, and what test vectors must be in place before cutover."
