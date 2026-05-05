# Canonical State Encoding for Replay

## Overview

**A snapshot is bytes. A state hash is a hash over bytes. Two snapshots that represent the same logical state but encode it as different bytes will hash differently — and the divergence protocol will scream about a "different state" that is the same state under a different encoding. The fix is canonical encoding: one logical state has exactly one byte representation, on every platform, in every language, in every library version.**

This sheet adapts canonical-encoding hygiene to replay-specific concerns: large-state efficiency, per-tick hashing, delta encoding, lazy fields, and the boundary with `axiom-audit-pipelines`'s canonical-encoding-for-fingerprinting sheet. Where the audit pack covers RFC 8785 / JCS for decision logging, this sheet covers state snapshots and the additional gotchas they bring. The deliverable is `11-canonical-state-encoding.md`.

## When to Use

Use this sheet when:

- `04-snapshot-strategy.md` calls for a state hash over the snapshot.
- The divergence protocol's compare-points hash state and the team is uncertain about hash stability.
- The system uses delta-encoded snapshots and the deltas must hash deterministically.
- A snapshot is read on a different machine / language / library version than where it was written.
- The system is writing state to a fingerprint chain (`axiom-audit-pipelines` overlap).

Do not use this sheet for:

- Decision-record canonical encoding for an audit trail — that is `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`. This sheet *cross-references* that sheet rather than duplicating it.
- Wire-format choice for inter-process communication (different problem; protobuf and msgpack are fine for non-deterministic transport).
- Storage layout (tiering, compression) — orthogonal to canonical encoding.

## Core Principle

> The encoding is part of the spec. The same state, encoded twice, must produce the same bytes — across machines, languages, library versions, today and after a refactor. Anything else is "the snapshot agrees with itself locally."

## What This Sheet Adds Over the Audit-Pack Sheet

The audit pack (`canonical-encoding-for-fingerprinting`) covers the foundational rules: RFC 8785 JCS, the `json.dumps()` failure modes, map ordering, float gotchas, library pinning, the chain-breaking event policy. Read it; do not duplicate it. This sheet's value is the *replay-specific delta*:

| Concern | Audit pack | This sheet |
|---------|-----------|-----------|
| Byte-stable encoding rule (JCS, sort keys, etc.) | Owns | References |
| `json.dumps()` failure modes | Owns | References |
| Float NaN/Inf, decimals as strings | Owns | References |
| Library version pinning | Owns | References |
| Per-tick hashing performance | — | Owns |
| Delta encoding for snapshots | — | Owns |
| Lazy / derived state | — | Owns |
| Pickle / cloudpickle as a non-canonical encoding | — | Owns |
| Tensor / array canonicalisation | — | Owns |
| Snapshot envelope schema | — | Owns |

A new system writing both decision logs and replay snapshots reads both sheets; the same canonical-encoding library configuration applies to both, and `02-canonical-encoding-spec.md` (audit) and `11-canonical-state-encoding.md` (this) cite each other.

## Per-Tick Hashing Performance

The divergence protocol (`05-`) emits a state hash at every compare-point. For a high-tick-rate simulation this can be 60+ hashes/sec over multi-megabyte state. Naive JCS-encode-then-hash is O(state size) per tick and cannot keep up.

Three patterns:

### Pattern A — Hash the canonical form once, hash deltas thereafter

```python
class IncrementalHasher:
    def __init__(self):
        self.cumulative = blake2b(digest_size=16)
        self.last = b""

    def initial(self, canonical_state_bytes: bytes) -> str:
        self.cumulative.update(canonical_state_bytes)
        self.last = canonical_state_bytes
        return self.cumulative.hexdigest()

    def update(self, delta_bytes: bytes) -> str:
        self.cumulative.update(delta_bytes)
        return self.cumulative.hexdigest()
```

The hash chains over canonical deltas; bit-identical for two replays *if and only if* the delta encoding is canonical. The delta encoding rule is part of `11-`.

### Pattern B — Hash structurally; trade per-tick cost for stability

A Merkle-tree hash over substructures. Each subsystem (env, policy, replay buffer) contributes a hash; the world hash is a hash of `(subsystem_name → subsystem_hash)` sorted by name. Per-tick cost = sum of changed subsystems' rehash, not the whole state.

### Pattern C — Hash a deterministic projection

If the full state is large but only a small projection matters for divergence (e.g., agent positions, not the rendering buffer), hash the projection. The projection is part of `11-`; changing the projection is a class-breaking event for the replay verification (the recorded hashes are tied to the projection, not the state).

Spec the pattern in `11-`. Mixing patterns across compare-points is allowed but each compare-point's pattern is fixed.

## Delta Encoding

If `04-` chooses delta-encoded snapshots, the delta itself must be canonical. A delta typically expresses: keys added, keys removed, keys whose value changed. Each axis is non-canonical by default:

```python
# NON-CANONICAL: insertion-order delta
delta = {"added": [k1, k3], "removed": [k7], "changed": {k4: v_new, k2: v_new}}

# CANONICAL: sort each axis by key, sort changed entries by key
delta = {
    "added": sorted([k1, k3]),
    "removed": sorted([k7]),
    "changed": dict(sorted({k4: v_new, k2: v_new}.items())),
}
```

The delta-canonicalisation rule is documented in `11-`. Two replays must produce byte-identical deltas at every step, or the chained hash diverges and the replay fails verification.

## Lazy and Derived State

State that is computed on first access (caches, memoised functions, JIT compilation results) is *not* part of the snapshot — but if it is hashed, the hash differs across runs that touched lazy fields in different order. Two rules:

1. **The snapshot's canonical form excludes lazy state.** If a cache is lazy, do not put it in the snapshot. Re-derive on rehydration.
2. **If lazy state must be in the snapshot for performance**, the snapshot rehydrates by *re-deriving* the lazy fields and overwriting whatever was stored, before the hash is computed. The stored bytes for lazy fields are advisory; the canonical form does not include them.

The trap: a derived field cached as a NumPy array, written to disk via `np.save`, read back via `np.load`. NumPy's binary format includes an alignment-padded header whose padding bytes are non-deterministic across versions. The fix: write derived fields with `tobytes()` only, after recomputing them; never trust derived bytes from disk.

## Pickle and CloudPickle Are Not Canonical

Pickle is the most-common non-canonical encoding in Python ML codebases. It is fast and easy and deeply non-canonical:

- Object identity affects the byte output (multiple references to the same object pickle differently).
- Class definitions are referenced by `module.qualname`; refactoring breaks pickle.
- Pickle protocol versions produce different bytes for the same object.
- CPython's pickle output depends on insertion order for `dict` and `set`.
- Numpy pickle includes endianness markers that vary across architectures.

The ban: **pickle is not a canonical encoding**. If pickle is in the snapshot path, the snapshot is not byte-stable. Use a structured encoding (JCS for non-numerical data; structured tensor encoding for arrays).

The exception: pickle is acceptable for *testing* state round-trips on the same machine and same Python version, when the property under test is "the state object survives serialisation," not "the snapshot's hash is stable."

## Tensor and Array Canonicalisation

Numerical arrays need their own canonicalisation rule. The audit pack's float-as-string rule does not scale to a tensor with 10M elements. Patterns:

| Tensor encoding | Determinism | Notes |
|-----------------|-------------|-------|
| `array.tobytes()` (raw IEEE 754 bytes) | Bit-stable on same architecture | Endianness varies; pin to little-endian |
| `np.save` (npy format) | Header padding non-deterministic | Avoid; or strip header |
| HDF5 with no compression | Compression varies; layout varies | Risky |
| Zarr / TileDB | Chunk layout non-deterministic by default | Configurable; must pin chunking |
| Custom: shape + dtype + raw little-endian bytes | Bit-stable | Recommended for snapshot bytes |

Recommended canonical form for a tensor:

```python
def canonical_tensor_bytes(t: np.ndarray) -> bytes:
    # Force little-endian, contiguous, known dtype
    if not t.flags["C_CONTIGUOUS"]:
        t = np.ascontiguousarray(t)
    t = t.astype(t.dtype.newbyteorder("<"), copy=False)
    header = struct.pack(
        "!B B 4s",
        len(t.shape),                         # rank (1 byte, ≤ 255 dims)
        t.dtype.itemsize,                     # element size (1 byte)
        t.dtype.kind.encode().ljust(4, b"\0") # dtype kind ('f', 'i', 'u', 'b', etc.)
    )
    shape_bytes = struct.pack(f"!{len(t.shape)}Q", *t.shape)
    return header + shape_bytes + t.tobytes()
```

The `!` prefix forces big-endian header (network byte order — fixed across architectures); the data is forced little-endian (pick one and hold). The header structure is part of `11-`.

## The Snapshot Envelope

A snapshot is not just the state; it carries metadata. The envelope is itself canonical:

```json
{
  "envelope_version": 1,
  "spec_ref": "11-canonical-state-encoding.md@v1.0.0",
  "tick_id": 12345,
  "code_version": "abc123",
  "config_hash": "def456",
  "seed_record": { "master": 42, "derivation_rule": "v1" },
  "subsystems": {
    "env":         "<canonical-tensor-bytes-encoded-as-base64>",
    "policy":      "<canonical-tensor-bytes-encoded-as-base64>",
    "buffer":      "<canonical-tensor-bytes-encoded-as-base64>"
  },
  "state_hash": "<hash over subsystems block, in canonical encoding>"
}
```

The envelope itself is JCS-canonical (audit pack rule); the inner tensor bytes are base64-encoded with no padding ambiguity. The `state_hash` field is the hash *of the canonical bytes of the subsystems block, before envelope serialisation* — not a hash of the envelope itself (which would chicken-and-egg).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `json.dumps(state)` for snapshot bytes | Use JCS-conformant library; pin version. |
| Pickle in the snapshot path | Replace with structured encoding. |
| `np.save` for tensors | Use canonical-tensor-bytes; strip headers. |
| Map ordering relies on Python `dict` insertion order | Sort keys before hashing. |
| Floats with NaN/Inf written to JSON | Reject upstream; or normalise per spec. |
| Tensor endianness not pinned | Force little-endian; document. |
| Lazy fields in canonical form | Exclude or re-derive before hash. |
| Delta key order varies across runs | Sort each delta axis. |
| Snapshot hash includes envelope metadata | Hash *contents* before envelope serialisation; the envelope nests the hash. |
| Library version not in `11-` | Pin library; CI runs RFC 8785 test vectors. |
| Cross-pack rule duplication (re-stating audit pack's gotchas) | Reference `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`; note where this sheet adds replay-specific deltas. |

## Spec Output (`11-canonical-state-encoding.md`)

The sheet's deliverable answers:

1. **Cross-link declaration** — cites `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` as the foundational rule; this sheet adds replay-specific deltas.
2. **Library pinning** — JCS library + version, BLAS / NumPy / tensor library + version (cross-link `08-floating-point-policy.md`).
3. **Per-tick hashing pattern** — A (incremental), B (Merkle), or C (projection); justification.
4. **Delta encoding rule** — if `04-` chooses deltas: how the delta is canonicalised (sort axes, key order, value encoding).
5. **Lazy / derived field policy** — what is excluded, what is re-derived, the order of re-derivation before hashing.
6. **Tensor canonicalisation** — header structure, byte order, contiguity rule, dtype encoding.
7. **Pickle policy** — banned in snapshot path; exceptions enumerated.
8. **Snapshot envelope schema** — fields, ordering, where the hash lives, how nested encodings combine.
9. **State hash function** — name, digest size, output encoding (hex, base64, raw).
10. **Class-breaking events** — encoding rule change, library upgrade, hash function change, envelope schema change, projection change.
11. **Test vectors** — at least one canonical-bytes test vector per state shape (small struct, large tensor, delta), checked in CI.

Without these eleven items the spec is incomplete and Check 15 (canonical state encoding) of the consistency gate will fail.

## Cross-Pack Notes

- `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` — foundational rules; this sheet references rather than duplicates.
- `floating-point-determinism.md` — the per-field float encoding ε is decided there; this sheet wires it into the hash.
- `04-snapshot-strategy.md` — chooses full vs delta vs event-sourced; this sheet defines the canonical bytes for whichever shape was chosen.
- `05-divergence-detection-and-localisation.md` — defines compare-points; this sheet defines the bytes hashed at each.

## The Bottom Line

**A snapshot is bytes; a state hash is a hash over bytes; the bytes must be the same across machines, languages, library versions, and refactors. The audit pack owns the foundational rules (JCS, library pinning, float gotchas); this sheet adds the replay-specific deltas (per-tick hashing performance, delta encoding canonicalisation, lazy fields, tensor canonical form, pickle ban, snapshot envelope). Cite the audit sheet rather than duplicate it; let the two specs evolve together.**
