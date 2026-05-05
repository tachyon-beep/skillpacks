# Fingerprint Chains and Integrity

## Overview

**A chain turns a sequence of independent entries into evidence about its own past. Tampering with any entry invalidates every later entry's hash. The cost is one extra hash field per entry; the property is non-repudiable order.**

This sheet specifies how to construct that chain, how to recover from gaps (lost or destroyed entries), how a verifier with only part of the trail can prove what they have, and what changes count as chain-breaking events.

Chain construction interacts with canonical encoding (`canonical-encoding-for-fingerprinting.md`) and signing (`signing-and-export-integrity.md`). Read those alongside this sheet.

## When to Use

Use this sheet when:

- Designing how entries are linked into a tamper-evident sequence.
- Choosing between linked-hash and Merkle-tree constructions.
- Designing for export of a subset of entries to a downstream verifier.
- Designing recovery procedures for damaged or lost segments of a trail.
- Producing `03-chain-and-integrity-spec.md`.

## Core Principle

> The chain is over canonical bytes, not over logical entries. The bytes are produced by the rule in `02-canonical-encoding-spec.md`. Changing the canonicalisation rule, the hash function, the chain construction, or the field set covered by the hash is a chain-breaking event — non-negotiable, regardless of how minor it seems.

## Two Constructions

### Linked-hash chain

Each entry includes `prev_hash`, the hash of the prior entry's canonical encoding. The chain head is the hash of the most recent entry; possessing the head, plus the canonical bytes of every entry, lets you verify the entire trail by walking forward and recomputing each `entry_hash`.

```
entry_0: { ..., prev_hash: null,           entry_hash: H(entry_0) }
entry_1: { ..., prev_hash: H(entry_0),     entry_hash: H(entry_1) }
entry_2: { ..., prev_hash: H(entry_1),     entry_hash: H(entry_2) }
...
```

Where `H(e)` is the hash of the canonical encoding of `e` excluding the `entry_hash` field (since `entry_hash` is computed last and stored, but it does not feed back into itself).

**Strengths:**

- Trivial to implement; one hash and one extra field per entry.
- Append-only with O(1) write cost.
- Single producer needs no coordination beyond knowing its own previous entry.
- A verifier with a head hash and the full trail proves order and integrity in one forward pass.

**Weaknesses:**

- Verifying a single entry from the middle requires the entire prefix from `entry_0`. Not friendly to partial export.
- Multiple producers contending for the head require synchronisation (whoever writes next must know the current head). At high concurrency this serialises.
- Branching is undefined; merging branches needs explicit policy.

**Use when:** there is one logical writer per chain, and verification typically processes the whole trail or contiguous segments.

### Merkle-tree chain

Entries are grouped into batches. Each batch is hashed into a Merkle tree; the tree root is the chain head for that batch. Batches are linked: the next batch's tree includes a reference to the prior batch's root.

```
batch 0:  e_0  e_1  e_2  e_3
                ↘   ↙   ↘   ↙
                  ↘    ↙
                   ROOT_0

batch 1:  e_4  e_5  e_6  e_7  ROOT_0_REF
                ...
                ROOT_1
```

A verifier with `ROOT_N` can verify any single entry in batch ≤N by walking O(log batch_size) tree nodes — the *Merkle inclusion proof* — without reading other entries.

**Strengths:**

- Verification of a single entry is O(log batch_size), not O(N).
- Exports of subsets carry small inclusion proofs; the recipient verifies against a known head without seeing other entries.
- High-concurrency producers can write into the same batch (the batch is sealed at boundaries; within a batch order is determined by tree position).
- Maps cleanly onto Certificate Transparency (RFC 6962) and similar transparency-log designs.

**Weaknesses:**

- More implementation surface (tree construction, sealing, root publication, inclusion-proof generation, consistency proofs across batches).
- Batch sealing introduces latency: an entry is not verifiable until its batch is sealed. Choose batch size against this latency target.
- Misuse of consistency proofs (proving that batch N+1 extends batch N rather than forking) is a frequent failure mode.

**Use when:** entries are exported in subsets to third-party verifiers, multiple producers write to the same chain, throughput is high, or a transparency-log model is wanted.

### Choosing

| Situation | Pick |
|----------|------|
| Single-writer, internal verification, end-to-end review the norm | Linked-hash |
| Multi-writer, partial export to third parties, transparency-log expectations | Merkle |
| Internal pipeline today, third-party export "someday" | Linked-hash now, with an explicit upgrade plan in `03-` — the upgrade is a chain-breaking event but smaller than retrofitting a third construction |
| Already integrated with an external transparency log | Mirror its construction; do not invent your own |

A pipeline can use linked-hash for one decision type and Merkle for another (different verifier expectations). Record the choice per decision type in `03-`.

## Hash Function

**Default: SHA-256** for new pipelines unless a specific reason names another. SHA-256 is the equilibrium choice — broadly available, not yet broken, performant, and what auditors expect.

Considerations:

- **SHA-3 (Keccak)** is a defensible alternative; recommended where SHA-256 collisions become a concern in the relevant time horizon (decades). Do not switch on speculation alone.
- **BLAKE3** is faster and arguably stronger but has a smaller installed base of audit tooling. Defensible for performance-critical internal pipelines; less defensible for trails an external auditor must verify with their tools.
- **MD5 and SHA-1 are forbidden.** No rationalisations. If you find them in legacy code, treat their replacement as a chain-breaking migration.

**Cryptographic agility:** the hash function name is part of `03-`. Plan now for the day SHA-256 becomes inadequate (post-quantum considerations notwithstanding — Grover's algorithm halves preimage difficulty, which means a 256-bit hash provides 128 bits of post-quantum strength, still well above the 128-bit security floor for most threat models). The migration plan, when needed, is a chain-breaking event that produces a transition note in `03-`: "entries before date D verify with hash H1; entries after verify with H2."

## What the Hash Covers

The `entry_hash` covers *all entry fields except `entry_hash` itself*, including:

- `prev_hash` (so the chain is self-referential)
- `output_hash` (so output redaction does not break the entry hash; see `retention-expiry-and-rtbf.md`)
- Any signature fields *if* the signature is over the same canonical bytes as the hash; otherwise, sign separately. See `signing-and-export-integrity.md` for the order-of-operations gotcha.

What the hash does *not* cover:

- Storage-layer framing (block headers, segment markers).
- Any field the schema marks as ignorable for integrity (rare; document explicitly).
- The hash of the entry itself (obvious, but worth stating because some implementations recursively include it and produce nonsense).

## Gap Recovery

Entries get lost. Disks fail. Backups corrupt. Operators delete by accident. A real audit pipeline plans for this *as a first-class capability*, not an exception.

The rule:

> A chain with a known gap is more trustworthy than a chain with an unknown gap. Make gaps visible.

### Detection

A gap exists when the next entry's `prev_hash` cannot be matched against any known entry. Detection is automatic on verification:

- For linked-hash: the verifier walks forward and finds an entry whose `prev_hash` doesn't match the previous entry's `entry_hash`. This is either a gap or a tamper. Distinguish by checking the *immediately prior* entry: if its `entry_hash` is correct against its own canonical bytes, the gap is upstream of it; otherwise, the prior entry is itself altered.
- For Merkle: a missing entry within a batch breaks the inclusion proof; a missing batch breaks the consistency proof against the next batch's root reference.

### Sealing the gap

When a gap is detected (and confirmed against backups, replicas, and the producer's records):

1. **Record a gap-marker entry** that names the gap explicitly. The marker includes:
   - `entry_id` (its own, time-ordered)
   - `gap_after_entry_hash`: the last known good entry hash before the gap
   - `gap_resumed_at_entry_hash`: the first known good entry hash after the gap (if known) or `null` if the gap extends to the present
   - `gap_cause`: structured cause code (`storage_corruption`, `accidental_delete`, `legal_redaction`, `unknown`)
   - `gap_evidence_ref`: pointer to the incident report, redaction order, or forensic findings
   - `prev_hash`: the hash of the gap-marker's *predecessor in the post-gap chain*, not the lost predecessor — the gap-marker is itself an entry in the new chain segment
2. **The chain after the gap continues with a new prefix.** The gap-marker's `entry_hash` becomes the `prev_hash` of the next ordinary entry. Verifiers walking the chain encounter the gap-marker and treat it as a documented discontinuity — they do not try to resolve `prev_hash` across the gap.
3. **The gap-marker is itself signed and persisted with the same discipline as ordinary entries.** A gap-marker that can be forged retroactively erases the trustworthiness of the discipline.

### What you must not do

- **Silent backfill.** Recomputing the chain by inserting recovered entries with newly-computed `prev_hash` values rewrites history. Forbidden.
- **"Best-effort" verification.** A verifier that ignores gaps it doesn't recognise is not a verifier.
- **Discarding the chain after a gap.** The pre-gap segment retains its integrity; the post-gap segment carries a new beginning. Both verify; the gap is a documented discontinuity, not a global failure.

## Partial-Trust Verification

A verifier may possess only part of a trail (a contiguous slice, a few selected entries, an export). What can they verify?

### Linked-hash

A contiguous slice from entry M to entry N verifies internally — re-hash each entry, check its `prev_hash` matches the prior. The slice cannot be tied to entries outside it without additional anchoring (a published head hash, a co-signed checkpoint).

**Anchoring mechanisms:**

- **Periodic head publication**: the producer publishes the chain head to a third party (a transparency log, a customer's system, a regulator's portal) at intervals. A slice between two published heads is bounded.
- **Notary co-signing**: a third party signs `(head_hash, time)` periodically. A slice that ends at or before a notarised head inherits the notary's signature.
- **External anchoring**: writing the head into a separate, independently-protected medium (a public blockchain, a write-once-read-many filesystem, a printed record). Pick anchoring mechanisms that an adversary cannot also alter.

A slice without an anchor proves only "these entries were in this order"; it cannot prove "this slice belongs to chain X". State this clearly in `03-`.

### Merkle

A subset export carries inclusion proofs against a published root. The verifier verifies each entry against the root via its inclusion proof, in O(log batch_size) per entry. Across batches, consistency proofs prove batch N+1 extends batch N rather than forking.

**Failure modes:**

- The root is not actually the canonical root the producer publishes — verifier reaches the wrong conclusion. Mitigation: the verifier obtains roots through a *different channel* than the entries (out-of-band, signed, cached publicly).
- Inclusion proof for an entry against a non-canonical root succeeds. Same mitigation.

### Honest scope

The verifier reports what they verified, not what they didn't. A status like "all 837 entries in the export verify against published root R_47, which was co-signed by notary N at 2026-01-15T08:00:00Z" is honest. "Trail verified" is dishonest because it implies the verifier checked anything outside the export.

## Chain-Breaking Events

A chain-breaking event is any change that means a verifier reading the chain end-to-end with a single rule will fail somewhere. Chain-breaking events are not failures — they happen, and they are recoverable — but they require explicit handling:

| Event | Required handling |
|-------|-------------------|
| Canonicalisation rule change | New `entry_version`; verifier loads version-specific rules; transition entry records the change |
| Hash function change | Boundary entry records the change with `prev_hash_v1` and `entry_hash_v2`; verifier swaps algorithms at the boundary |
| Chain construction change (linked-hash → Merkle) | Effectively a new chain that anchors to the old chain's terminal head; transition documented |
| Field-set covered by hash changes | Same as canonicalisation rule change |
| Signature scheme change (separate from hash) | Documented in `04-signing-and-export-spec.md`; chain-side unaffected if hashing is unchanged |
| Producer identity change at scale | Not a chain-breaking event; new entries simply have new `producer_id` |
| Storage migration | Not a chain-breaking event if entries are byte-identical after migration; verify by re-hashing |
| Restoration from backup | Not chain-breaking; verifier re-hashes restored entries against their original canonical bytes |

The list of chain-breaking events lives in `03-chain-and-integrity-spec.md`. A change not on the list is treated as chain-breaking by default.

## Clock Dependence

The chain proves *order*. It does not prove *time*. `decided_at` is a producer-asserted timestamp; an adversary with control of the producer can write any timestamp they like. Ordering is non-repudiable; absolute time is not.

To bound time:

- **Periodic notary co-signing** (above) — a notarised head at time T proves all entries before it are at-or-before T.
- **External time anchors** — committing the chain head to a system whose timestamps you trust independently (a public ledger, a TSA per RFC 3161). The anchor proves "head H existed by time T".
- **Producer-side TSA tokens** — RFC 3161 timestamping authority tokens attached to entries. Adds independent time evidence at per-entry cost.

State the time-trust model in `03-`. "We trust producer timestamps" is a valid choice for tier S; it is not a valid choice for tier L without supporting evidence.

## Implementation Discipline

### `entry_hash` is computed once and stored

Recomputing `entry_hash` on every read is correct but wasteful. Store it. Verifiers re-hash the canonical bytes on verification, not on read.

### `prev_hash` is the predecessor's stored `entry_hash`

Not "the hash of the predecessor", which leaves room for ambiguity (hash before or after stored?). The discipline: write order is *(1) construct entry without `prev_hash` and `entry_hash`*, *(2) populate `prev_hash` from the predecessor's stored value*, *(3) canonical-encode*, *(4) hash to produce `entry_hash`*, *(5) persist*.

### Verifier and producer share the same canonicaliser version

A verifier on a different library version is a different system. Pin both. CI test vectors apply on both sides.

### Failures during write must not produce orphan entries

If the chain head update succeeds but the entry write fails, the chain is broken. Use whatever atomicity the storage layer provides (transactions, write-then-publish-head, two-phase commit) to ensure entry-and-head update together. If the storage layer does not support atomicity, this becomes a discipline issue documented in `06-storage-and-retention.md`.

### A chain has a beginning

The first entry's `prev_hash` is `null` by convention. The producer signs and publishes that first entry's `entry_hash` as the *chain anchor*. Verifiers receive the anchor through an out-of-band trusted channel and use it as the verification root.

A chain without an anchor verifies that "these entries form a chain", not "these entries form *the* chain". For audit purposes, the latter matters.

## Spec Output (`03-chain-and-integrity-spec.md`)

The sheet's deliverable answers, in order:

1. **Construction chosen** — linked-hash or Merkle, per decision type if mixed.
2. **Hash function** — name, output length, library + version per language.
3. **Hash coverage** — which fields, in which canonicalisation rule version.
4. **Anchor** — how the chain's first entry hash is published; through which channel.
5. **Periodic anchoring / notarisation** — schedule, mechanism, channel, notary identity.
6. **Gap recovery procedure** — gap-marker entry shape, sealing protocol, evidence requirements.
7. **Partial-trust verification model** — what a verifier with N entries plus an anchor can prove.
8. **Time-trust model** — producer-only, notary-anchored, external-time-anchored.
9. **Chain-breaking-event list** — explicit enumeration; the catch-all "anything else not on this list is chain-breaking by default" clause.
10. **Cryptographic agility plan** — when to migrate the hash function; how the migration produces a transition entry; how legacy entries continue to verify.

Without these ten items the spec is incomplete and Check 4 of the consistency gate will fail.

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| `prev_hash` computed lazily on read | Different bytes on different reads; chain unverifiable | Compute on write; store; re-verify by re-hashing canonical bytes |
| Silent backfill of recovered entries | Rewrites history; the cure is worse than the disease | Gap-marker entry; the chain after the gap is a new segment |
| "We use SHA-256 *plus* a salt" | Salts in audit chains break inter-organisation verification | SHA-256 over canonical bytes, no salt; if confidentiality matters, encrypt at rest separately |
| Hash covers "the important fields" | Adversary mutates the rest; chain validates the lie | Hash covers everything except `entry_hash` itself |
| Verifier ignores gap-markers it doesn't recognise | Gap policy is the verification | Verifier rejects unknown gap-cause codes; updates verifier or rejects trail |
| Transparency-log style claims without consistency proofs | Producer can fork the chain; adversary follows | Implement consistency proofs; verifier checks them |
| Chain without an anchor | Entries form *a* chain, not *the* chain | Publish first entry's hash through an independent channel |
| Producer trusts its own clock at tier L | Time-shift attack is undetected | Add notary co-signing or external time anchoring |
| MD5/SHA-1 in legacy code "left alone for compatibility" | Collision attacks are real and decades old | Migrate as a chain-breaking event; document transition |

## The Bottom Line

**Hash canonical bytes with SHA-256. Pick linked-hash for single-writer pipelines and Merkle for partial-export ones. Make gaps visible with first-class gap-marker entries. Anchor the chain externally so a verifier knows it has *the* chain, not *a* chain. Treat the chain construction, hash function, canonical encoding rule, and field coverage as a unit — change any one and you have a chain-breaking event with a documented transition.**

---

**Retrieval test (run at end of build):** "Storage corruption silently flipped two bits in entry 4,892 of a linked-hash chain. The corruption is detected on routine verification three months later. Walk through what the verifier sees, how to distinguish corruption from tamper, and what the recovery looks like."
