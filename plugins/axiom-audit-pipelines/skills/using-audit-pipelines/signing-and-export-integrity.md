---
name: signing-and-export-integrity
description: Use when designing the signature scheme, key-rotation discipline, signature granularity (per-entry / per-batch / per-export), and the protocol for exporting a subset of entries to a third-party verifier without breaking proofs over the rest of the trail. Produces `04-signing-and-export-spec.md`.
---

# Signing and Export Integrity

## Overview

**Hashes prove the bytes weren't altered. Signatures prove who produced them. Exports prove what a third party can verify standalone, without re-trusting the producer. The three are separate properties; conflating them is how exports break in production.**

This sheet specifies the signature scheme, key-rotation discipline, signature granularity (per-entry, per-batch, per-export), and the protocol for exporting a subset of entries to a downstream verifier without breaking proofs over the rest of the trail.

Read alongside `fingerprint-chains-and-integrity.md` (chain construction is upstream of signing) and `canonical-encoding-for-fingerprinting.md` (the bytes being signed).

## When to Use

Use this sheet when:

- Designing the signature layer over an already-specified chain.
- Choosing between symmetric (HMAC) and asymmetric (Ed25519, ECDSA P-256, RSA-PSS) signatures.
- Designing key rotation, key custody, key compromise response.
- Planning subset exports to auditors, regulators, customers, or downstream systems.
- Producing `04-signing-and-export-spec.md`.

## Core Principle

> Signatures bind producer identity to canonical bytes. The bytes being signed must be the *exact* canonical encoding the chain hashes — same canonicaliser version, same field set, same byte order. Anything else creates a class of attacks where the chain validates but the signature does not, or worse, the signature validates over different bytes than the chain.

## Symmetric vs Asymmetric

### HMAC (symmetric)

A keyed hash over canonical bytes; the verifier holds the same key as the producer.

**Strengths:**

- Cheap (one HMAC per signature; ~hash cost).
- Operationally simple where producer and verifier are the same organisation.
- No PKI to manage.

**Weaknesses:**

- The verifier could have produced the signature too. **HMAC does not prove producer identity to a third party.** It proves "either the producer signed this, or someone with the key signed this".
- Key rotation is coordinated: every verifier needs the new key.
- Cannot meaningfully sign third-party-verifiable exports.

**Use when:** producer and verifier share a trust zone, signatures are an integrity check rather than non-repudiation evidence.

### Asymmetric (Ed25519 default)

Producer holds a private signing key; verifiers hold the corresponding public key.

**Strengths:**

- Non-repudiable by a third party who has only the public key.
- Public-key distribution is easier than secret-key distribution.
- Key rotation does not require coordinated secret distribution.

**Weaknesses:**

- More cost per signature (Ed25519 ~ tens of microseconds; ECDSA more).
- More implementation surface (key generation, custody, rotation, revocation, public-key publication).

**Use when:** any verifier outside the producer's trust zone, regulatory non-repudiation requirement, court-evidence quality.

**Default scheme: Ed25519 (RFC 8032)** for new pipelines. Ed25519 is the equilibrium choice — fast, deterministic (no per-signature randomness to leak), small keys (32 bytes public, 32 private, 64 signature), broad library support. ECDSA P-256 is acceptable where Ed25519 is not available; RSA-PSS is acceptable where ECDSA is not. Plain ECDSA without RFC 6979 deterministic-k is not acceptable.

A pipeline can use HMAC internally and asymmetric for export — record the dual scheme in `04-` explicitly.

## What Gets Signed

The signature covers the **canonical bytes the chain hashes**. Concretely:

- For per-entry signing: the canonical encoding of the entry, *excluding* the `signature` and `entry_hash` fields. The verifier reconstructs the canonical bytes, computes `entry_hash`, then verifies the signature against the same canonical bytes (or against `entry_hash` directly, if the scheme commits to the hash).
- For batch / Merkle-root signing: the canonical encoding of the published root structure (root hash, batch interval, producer identity, timestamp).

**The order-of-operations gotcha:** if `entry_hash` is computed *before* `signature` is populated and `signature` is then included in the entry, two scheme designs are possible:

1. **Sign-then-hash:** signature is computed first, then `entry_hash` covers everything including the signature. Verifier checks `entry_hash` first; signature verification is implicit (any signature change breaks the hash). Simple, but a forged hash can hide an unsigned entry — only mitigated by also requiring an explicit signature check.
2. **Hash-then-sign:** `entry_hash` is computed first, signature is computed over `entry_hash`. The signature itself is *not* covered by `entry_hash`. Verifier independently checks both. Standard for transparency-log designs.

Pick one in `04-`, document it, and never mix. **Mixed entries break verifiers.**

## Signature Granularity

| Granularity | What's signed | Cost | When to choose |
|-------------|---------------|------|----------------|
| Per-entry | Each entry individually | High (one signature per entry) | Tier L+, regulatory non-repudiation, single-entry exports common |
| Per-batch | The Merkle root of a batch | Low (one signature per batch) | High-throughput pipelines, transparency-log model, exports as batch + inclusion proofs |
| Per-export | A wrapper structure naming the entries in the export | Lowest amortised | Subset exports to a single audience, signed envelope per export |
| Hybrid | Per-batch routinely + per-export envelope on demand | Moderate | Pipelines that mostly verify in-place but occasionally export |

Per-entry signing for tier L+ is the safest default for evidence quality but costs throughput. Per-batch with per-export envelopes is the equilibrium for high-throughput pipelines. Decide per decision type if the pipeline has mixed needs.

## Key Custody

A signing key in a config file or an environment variable is not custody; it is a leak waiting for a deploy diff. Three custody models, in increasing order of strength:

1. **Application-managed key in encrypted storage.** Key encrypted at rest with a KEK held by a key-management service (KMS). Application decrypts in-process. Cheap; vulnerable to memory disclosure and process-compromise.
2. **KMS-issued sign-on-demand (cloud KMS, HashiCorp Vault Transit).** The application sends bytes; the KMS returns a signature. Key never leaves the KMS. Rate-limited; cost per signature is nontrivial. Standard for cloud-native pipelines.
3. **Hardware Security Module (HSM, FIPS 140-3 Level 2/3).** Key generated and used inside tamper-resistant hardware. Higher cost, lower throughput, audit-friendly. Required by some regulators.

Tier L expects (2) at minimum; tier XL typically requires (3) or a cloud KMS HSM offering with FIPS 140-3 Level 3 attestation.

## Key Rotation

Keys must rotate. Rotation policy in `04-` answers:

1. **Cadence.** Routine rotation interval (typically 1 year for production signing keys; shorter for HMAC).
2. **Trigger conditions.** Suspected compromise, personnel change, KMS migration, scheduled cadence.
3. **Identifier.** Each key has a `key_id`; signatures carry `key_id` so the verifier knows which public key to load. Without `key_id`, the verifier cannot rotate.
4. **Overlap window.** During rotation, both old and new keys are valid for some interval; producers cut over at a defined moment (the "key-rotation entry"); verifiers accept either key for entries in the window.
5. **Public-key publication.** Where the verifier obtains the public key. A key-rotation event publishes the new public key through the same channel as the old. The publication itself is signed by a *higher-trust* key (an organisational signing key, an HSM root) — otherwise an adversary who controls publication can rotate to their own key.
6. **Compromise response.** What happens to entries signed with a compromised key — the response is in `07-threat-model.md`, but the operational mechanics live here: revocation list, re-signature with a new key, scope of re-verification.

### The key-rotation entry

A key rotation produces a structured entry of its own, in the chain:

```
{
  decision_type: "audit.key-rotation",
  entry_version: ...,
  decided_at: ...,
  output: {
    old_key_id: "K-7",
    new_key_id: "K-8",
    rotation_reason: "scheduled" | "compromise" | "personnel-change" | ...,
    rotation_authority: <signed-by-higher-trust-key>
  },
  prev_hash: ...,
  entry_hash: ...
}
```

The rotation entry is itself signed by both the old and new keys (or by the rotation authority, depending on the trust model). Verifiers walking the chain use the rotation entry to swap public keys at the boundary.

## Compromise Response

When a private key is compromised:

1. **Stop signing with the compromised key immediately.** Producer cuts over to a fresh key via the rotation procedure.
2. **Publish a revocation entry** (similar shape to the rotation entry, with `compromise: true`) signed by the rotation authority (not by the compromised key).
3. **Define the affected window.** Entries signed by the compromised key between the suspected compromise time and the revocation are *suspect*. They are not automatically invalid — the chain still verifies bytes — but their signatures alone are not sufficient evidence of producer identity.
4. **Re-sign or co-sign the affected window with the new key** if the chain construction allows, or accept the suspect window with cross-evidence (timestamps from external anchors, third-party witnesses, internal logs that can corroborate identity).
5. **Update the threat model** in `07-` with the incident and the residual risk.

A compromise without a revocation entry is a worse failure than the compromise itself, because verifiers downstream cannot tell affected entries from unaffected.

## Subset Export

Exporting a subset of entries to a third party — auditor, regulator, customer with right-of-access — without sharing the full trail:

### What the export envelope contains

```
export-envelope {
  export_id: <uuid>,
  exported_at: <RFC 3339 UTC>,
  exporter: <producer or signing authority>,
  scope: {
    decision_types: [...],
    time_range: { from, to },
    filter_criteria: <as-applied>
  },
  entries: [<canonical bytes of each exported entry>],
  inclusion_proofs: [<per entry, against published root>],   // Merkle case
  chain_anchor: { head_at_export: <hex>, anchor_signature: <bytes> },  // linked-hash case
  envelope_signature: <signature over the canonical encoding of everything above except envelope_signature>,
  envelope_signing_key_id: <key_id>
}
```

Choosing what the envelope binds:

- For a Merkle chain: each exported entry carries an inclusion proof against a published root. The envelope's signature binds the export, but the entries can be verified directly against the root through their proofs without further reference to the envelope. **This is the strongest model for third-party verifiability.**
- For a linked-hash chain: the export is a contiguous slice from entry M to entry N, plus an anchor — typically the head hash at export time, signed by a notary or a higher-trust key. The verifier verifies the slice internally and trusts the anchor. **Subsets that aren't contiguous slices in linked-hash chains cannot be exported without losing the chain property** (unless a separate per-entry signature is added, which is the per-entry signing case).

### Common export gotchas

| Gotcha | What goes wrong | Fix |
|--------|-----------------|-----|
| Subset is non-contiguous slice in a linked-hash chain | Verifier cannot reconstruct `prev_hash` chain | Use Merkle, or add per-entry signatures, or export contiguously |
| Envelope signed but inclusion proofs absent | Verifier trusts the envelope but cannot verify entries against any anchor | Inclusion proofs are mandatory for Merkle exports |
| Anchor unsigned by an authority external to producer | Producer can rotate keys retroactively to favour their position | Anchor is signed by an authority *outside* the producer's signing path |
| Filter criteria included in envelope but not over the canonical bytes | Adversary modifies criteria after the fact | Filter criteria are part of the envelope's signed body |
| Time range claimed in envelope doesn't match `decided_at` of entries | Verifier accepts mismatch | Verifier validates time-range against `decided_at` of every entry |
| Export "selects" entries by mutable predicate | The same predicate evaluated later might select different entries | Predicates are evaluated once, the resulting set is fixed in the envelope |

## Cryptographic Agility

Signature schemes change. RSA was once standard; ECDSA replaced it for many uses; Ed25519 is the modern default; post-quantum signature schemes (ML-DSA / Dilithium per FIPS 204) are coming. Plan now:

- **`signature_scheme` field** in entries names the scheme (`ed25519`, `ecdsa-p256`, `rsa-pss-sha256`, `ml-dsa-65`). Verifier reads the scheme and selects the algorithm.
- **Migration plan in `04-`** names: when the migration would happen (regulatory deadline, library availability, threat-model trigger), what the transition entry looks like, how legacy entries continue to verify with the old scheme.
- **Test vectors** for the chosen scheme, kept in CI alongside canonicalisation vectors.

A migration that doesn't preserve verifiability of pre-migration entries is a re-signing event, not a migration. Re-signing is permissible (you produce additional signatures with the new key alongside the old) but expensive at scale; document the policy in `04-`.

## Spec Output (`04-signing-and-export-spec.md`)

The sheet's deliverable answers, in order:

1. **Scheme** — HMAC, Ed25519, ECDSA P-256, RSA-PSS, with rationale for non-default choices.
2. **Granularity** — per-entry, per-batch, per-export, hybrid.
3. **What is signed** — exact canonical-byte specification, sign-then-hash vs hash-then-sign.
4. **Key custody** — application-managed, KMS-issued, HSM, with FIPS level if applicable.
5. **Key rotation** — cadence, triggers, `key_id` policy, overlap window, publication channel.
6. **Compromise response** — runbook reference, revocation-entry shape, affected-window policy.
7. **Export envelope** — shape, fields signed, anchor mechanism, inclusion-proof strategy.
8. **Subset-export rules** — contiguous-only or arbitrary-subset, per chain construction.
9. **Cryptographic agility plan** — migration timeline, transition-entry shape, re-signing policy.
10. **Cross-pack handoff** — where in `ordis-security-architect`'s control set the signing infrastructure lives (KMS, HSM, key-publication endpoint).

Without these ten items the spec is incomplete and Check 4 of the consistency gate may fail (chain-and-signing consistency).

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| HMAC for third-party verification | Verifier could have produced the signature; non-repudiation absent | Asymmetric (Ed25519 default) for any cross-trust-zone verifier |
| Signature over different bytes than the hash | Chain validates while signature doesn't (or worse, vice versa) | Sign exactly the canonical bytes the hash covers |
| `key_id` absent from signatures | Cannot rotate keys; verifier cannot pick the right public key | `key_id` is mandatory in every signature |
| Public-key publication signed by the same key it publishes | Adversary controlling publication can rotate to their own key | Higher-trust key (organisational, HSM root) signs publications |
| Compromised key entries silently treated as valid forever | Attacker forges entries after compromise window | Revocation entry plus affected-window policy |
| Non-contiguous linked-hash exports | Verifier cannot link the entries | Per-entry signatures or contiguous-only or migrate to Merkle |
| Inclusion proofs computed against producer-supplied root | No third-party anchor; verifier circular-trusts producer | Root anchored externally (notary, public ledger, regulator portal) |
| ECDSA without RFC 6979 deterministic-k | Per-signature randomness leak risk | Use Ed25519 instead, or ECDSA-RFC6979 |
| Plain ECDSA P-256 over a SHA-256 hash, with the hash output truncated | ECDSA over short hashes weakens the security | Hash output ≥ 256 bits for 128-bit security floor |
| Signing scheme migration treated as "drop-in" | Verifiers fail on the boundary | Transition entry; test vectors; migration plan in `04-` |

## The Bottom Line

**Asymmetric signatures (Ed25519) for any cross-trust-zone verifier; HMAC only intra-zone. Sign exactly the canonical bytes the chain hashes. `key_id` mandatory. Rotation cadence published, rotation entries chained, public-key publications signed by a higher-trust key. Subset exports carry inclusion proofs (Merkle) or are contiguous slices with external anchors (linked-hash). Compromise produces a revocation entry, an affected window, and a documented response — silence on compromise is a worse failure than the compromise itself.**

---

**Retrieval test (run at end of build):** "We need to send a year's worth of one decision_type's entries to a regulator. Our chain is linked-hash. What does the export look like, and what's the failure mode if we try to send a non-contiguous subset?"
