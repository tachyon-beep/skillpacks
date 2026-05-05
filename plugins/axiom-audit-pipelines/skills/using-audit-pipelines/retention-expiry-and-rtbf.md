---
name: retention-expiry-and-rtbf
description: Use when reconciling append-only storage with retention obligations, right-to-be-forgotten, redaction-for-legal-hold, and incident-investigation preservation. Mechanisms: cryptographic erasure, redaction-with-witness, segregated-PII architectures, explicit waivers anchored in named legal authority. Pairs with immutable-storage-patterns. Contributes to `06-storage-and-retention.md`.
---

# Retention, Expiry, and Right-to-be-Forgotten

## Overview

**Append-only storage and right-to-be-forgotten look like a contradiction — and they are, if you treat them as such. The reconciliation is not "exception handling"; it is a primary design property of the trail, made explicit in the schema and the threat model.**

This sheet specifies how a chain that must "never lose data" interacts with retention obligations that say "destroy this after N years," "delete this on data-subject request," "redact this for legal hold," and "preserve this for an incident investigation despite ordinary retention." The mechanisms are cryptographic erasure, redaction-with-witness, segregated-PII architectures, and explicit waivers anchored in legal authority. None of them are silent; all of them are recorded.

## When to Use

Use this sheet when:

- A regulator imposes retention requirements (minimum or maximum).
- A regulation imposes RTBF obligations (GDPR Article 17, CCPA right to delete, sectoral data-protection laws).
- A legal hold or e-discovery order intersects with the trail.
- An incident response requires a retention extension on subset entries.
- Producing the retention portion of `06-storage-and-retention.md`.

## Core Principle

> The trail's append-only property is preserved at the *chain* layer. Retention is enforced at the *content* layer. Cryptographic erasure destroys the meaning of an entry without removing the entry; a redaction-with-witness keeps the chain whole while marking the content gone. Right-to-be-forgotten is implemented without breaking the chain because the chain is over hashes, not over plaintext.

## The Reconciliation

Three properties are simultaneously required:

1. **Chain integrity.** Every entry's hash matches; chain links are unbroken.
2. **Retention.** Some content must be preserved; some must be destroyed; some moves between states.
3. **Right-to-be-forgotten.** A data subject's PII must be unrecoverable on request.

These reconcile because *the chain covers `entry_hash`*, and `entry_hash` covers (among other things) `output_hash` and `inputs_commitment.hash` — both of which can survive when the underlying content is destroyed.

| Layer | What persists forever | What can be destroyed |
|-------|------------------------|------------------------|
| Chain | `entry_id`, `prev_hash`, `entry_hash`, `decided_at`, `decision_type`, `producer_id` | — |
| Schema | Field structure, types, validation rules | — |
| Content | `inputs_commitment.hash`, `output_hash` | `inputs_commitment.inline` (if present), `output` plaintext, `inputs_commitment.ref`'s referent |
| PII registry (separate) | — | The plaintext PII bound to the entry |

The chain proves "a decision of type T occurred at time D, with inputs hashing to H1 and output hashing to H2." That fact is permanent. The plaintext of inputs and output may be destroyed without violating the chain.

## Cryptographic Erasure

Encrypt at content level with a per-record (or per-data-subject) key held in a key registry. Destroying the key destroys the meaning of the encrypted content; the ciphertext remains on disk but is unrecoverable.

Mechanism:

```
Entry storage:
  entry { ..., output: <ciphertext_E_K(canonical output)>, output_hash: H(canonical output), ... }
  inputs_commitment.inline: <ciphertext_E_K(canonical inputs)>
  inputs_commitment.hash: H(canonical inputs)

Key registry (separate store):
  K_for_entry_e: <symmetric key>
  K_for_subject_s: <subject-level key, used for all entries about subject s>
```

When retention/RTBF triggers destruction, the key is destroyed (overwritten in the registry, removed from KMS). The entry survives; its ciphertext is now opaque; its hashes still verify; the chain is intact.

**Strengths:**

- Append-only is preserved exactly — nothing is removed from the trail store.
- Chain integrity unaffected.
- Bulk RTBF is fast — destroy one subject-level key, all that subject's entries become unrecoverable.
- Key destruction is a discrete, auditable event (it is itself a decision in the audit pipeline).

**Weaknesses:**

- Requires planning at write time (choosing key granularity).
- A key registry is now a critical component with its own threat model.
- "Destroyed" is a property of the registry; an attacker who has copies of past key material defeats the erasure for entries they've already exfiltrated.
- Subject-level keys couple multiple entries' erasure together; per-record keys are flexible but more expensive.
- Quantum cryptanalysis future-risk: AES-256 is comfortably post-quantum-strong, but pre-AES-256 ciphers are not. State the cipher choice.

**Use when:** RTBF is a real obligation, the trail will hold any plaintext PII, and the key registry can be operated with its own discipline.

### Key granularity decisions

| Granularity | Use case |
|-------------|----------|
| Per-entry | Maximum flexibility, but expensive at scale; rarely needed |
| Per-data-subject | Standard for RTBF; one key destruction handles all of a subject's entries |
| Per-decision-type | Coarser; supports retention-class-based destruction; less RTBF-friendly |
| Per-time-bucket | Supports temporal retention (destroy all 2020 keys); not RTBF-friendly |

A hybrid is common: per-data-subject keys for entries containing PII, per-time-bucket for entries without. State the scheme in `06-`.

## Redaction-with-Witness

For content that must be destroyed but where cryptographic erasure isn't available (legacy entries, content held by external custodians, content where keys were never separated), redaction-with-witness produces a *successor entry* that records the destruction:

```
Original entry:    e_n { decision_type, output: <plaintext>, ... }
                       (entry_hash captured in chain)

Redaction event:   e_m { decision_type: "audit.redaction",
                         output: { redacted_entry_id: e_n.entry_id,
                                   redaction_reason: "rtbf-request" | "legal-redaction" | ...,
                                   redaction_authority: <signed reference>,
                                   redacted_at: ... },
                         prev_hash: ...,
                         entry_hash: ... }

Original entry's content:  destroyed in storage; the row/object retains
                            entry_hash and prev_hash (chain bytes) but the
                            content fields are zeroed or set to <redacted>.
```

After redaction, the original entry's `entry_hash` *no longer matches* the entry's stored content (because the content is gone). This is detectable by a verifier — and intentionally so. The verifier sees the original entry, sees its content is `<redacted>`, walks forward, finds the redaction-event entry that names it as the predecessor, and concludes: "this entry was redacted on date D under authority A; the chain integrity is preserved at the *hash* level (because the chain bytes are unchanged) but the content is gone by design."

**Strengths:**

- Works for legacy content where cryptographic erasure wasn't planned.
- Explicit, witnessed destruction.

**Weaknesses:**

- The original entry's `entry_hash` cannot be re-verified against content — only the absence is verifiable.
- Verifier tooling must understand redaction events; naive verifiers see broken hashes.
- A malicious "redaction" can hide tamper; the redaction-authority signature is the load-bearing control.

**Use when:** cryptographic erasure isn't possible (legacy content, external custodians); the redaction authority can be made auditable.

## Segregated PII Architecture

Best-effort prevention rather than reactive destruction: keep PII out of the audit pipeline in the first place.

```
Decision producer → audit pipeline    (entries reference subjects by opaque id)
                                      (no plaintext PII; chain bytes are PII-free)

PII registry      → separate system   (subject_id → plaintext PII)
                                      (RTBF destroys PII registry rows;
                                       audit pipeline is unaffected;
                                       audit entries reference subject_ids
                                       that resolve to "unknown subject" after RTBF)
```

The audit pipeline is forever; the PII registry is short-lived. Audit entries hold opaque ids; querying a redacted subject_id returns "unknown subject"; the audit chain remains complete and verifiable.

**Strengths:**

- The audit pipeline never holds PII; RTBF doesn't touch the chain.
- Cleanest reconciliation; no exotic primitives.
- Most defensible at audit-of-the-audit-system review.

**Weaknesses:**

- Requires the system to be designed with this separation from the start.
- Joins between audit and PII for legitimate purposes (investigation) require explicit access flow.
- Doesn't help when the regulator demands the PII *be in* the audit log.

**Use when:** designing a new system; can isolate PII into a separate, mutable store; chain integrity must be airtight.

This is the *preferred* architecture for new pipelines where the regulatory environment permits.

## Legal-Authority-Recorded Waivers

Some destructions are required by named legal orders that cannot be implemented via the technical mechanisms above (e.g., a court order to destroy specific entries with cryptographic completeness, including any encrypted ciphertext). The mechanism:

1. The destruction is performed using whatever mechanism the order requires.
2. A waiver entry is written into the chain naming: the legal authority (court name, case number, order date), the affected entry IDs (or the criteria), the date of execution, the operator who executed.
3. The waiver entry is signed by the highest-trust signing key available (HSM, organisational signing key — not the routine producer key).
4. The waiver is itself published to any external anchor service (notary, public ledger), separately from ordinary chain anchoring.

A waiver is the explicit acknowledgement that retention discipline was overridden by external authority. It is auditable; a waiver without a named authority is forgery.

**Use when:** a court order, regulator directive, or unbreakable legal compulsion requires destruction beyond what the standard mechanisms achieve.

## Retention Policy Components

The retention portion of `06-storage-and-retention.md` answers, per `decision_type`:

1. **Minimum retention.** The shorter of (a) regulatory minimum, (b) contractual minimum, (c) operational minimum (incident investigation, complaint window). Pick the longest.
2. **Maximum retention.** Where regulators or data-protection law impose a maximum (GDPR's storage-limitation principle), the date beyond which content must be destroyed.
3. **Trigger conditions for destruction.** Time-based (after N years from `decided_at`), event-based (after closure of related case), or request-based (RTBF, data-subject request).
4. **Mechanism per decision_type.** Cryptographic erasure (and key granularity), redaction-with-witness, segregated-PII, legal-authority waiver.
5. **Legal-hold capability.** How a hold is placed (suspending the retention timer for a specific scope), how it's lifted, who can authorise.
6. **Cross-pack handoff.** Compliance / legal / privacy team interfaces; how they request destructions or holds; how the system records their authority.

## Retention Decision Tree

```
Is the content in the audit pipeline genuinely necessary?
├─ No → segregate; keep PII out; audit holds opaque ids only
└─ Yes → continue

Is RTBF a real obligation for this content?
├─ Yes → cryptographic erasure with subject-level keys
│        OR redaction-with-witness if encryption-at-write was not planned
└─ No → continue

Is there a maximum retention that applies?
├─ Yes → schedule destruction; cryptographic erasure or redaction-with-witness at expiry
└─ No → indefinite retention with the explicit case for indefiniteness in 06-

Are there special destruction orders foreseen (court, regulator)?
├─ Yes → legal-authority-recorded waiver protocol documented
└─ No → no waiver protocol needed; revisit if circumstances change
```

## Retention as Audit Events

Every destruction, redaction, key-rotation-for-erasure, or legal-hold action is itself a decision that goes into the audit pipeline:

- `audit.key-destruction` — entry naming the key id, the affected scope, the authority
- `audit.redaction` — as above
- `audit.legal-hold-placed` / `audit.legal-hold-lifted` — naming scope, authority, duration
- `audit.retention-expiry` — scheduled destruction by retention policy

The audit pipeline thus self-references: it carries records of its own destructions. This creates a partial recursion (the audit-of-audit-actions can grow), and the schema in `01-` should treat these as ordinary decision types — `decision_type: "audit.*"` — with their own retention, which is typically *longer* than the entries they govern (the destruction record outlives what was destroyed, so future auditors can see what was destroyed and why).

## Spec Output (retention portion of `06-storage-and-retention.md`)

Answers, in order:

1. **Per-`decision_type` retention table.** Min retention, max retention, mechanism, key granularity if cryptographic erasure.
2. **RTBF protocol.** Per regulation (GDPR, CCPA, sectoral). Triggering, processing, completion, evidence of completion.
3. **Legal-hold protocol.** Trigger, scope, suspension behaviour, lift, evidence.
4. **Destruction-as-audit-event.** Schema for audit.* destruction entries; their retention (typically permanent at L+).
5. **Mechanism-by-mechanism playbook.** Concrete steps for each RTBF / retention action, runbook references.
6. **Cross-pack handoff.** Compliance team's interface; legal team's interface; how their requests become audit-pipeline entries.
7. **Architecture decision.** Which mechanism is dominant (segregated-PII preferred for new builds); why deviations exist for legacy.

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| RTBF "violates audit"; treat as exception | Audit fails legal review; system is not deployable in EU/UK | Cryptographic erasure or segregated-PII as primary design |
| Plaintext PII inlined in entries "for convenience" | RTBF requires destroying entries; chain breaks | Segregate PII; reference by opaque id |
| Cryptographic erasure with one global key | Destroying it nukes the entire trail; never used | Per-subject or per-bucket keys |
| Redaction silently overwrites content with no chain-side record | Indistinguishable from tamper | Redaction-event entry; redacted entry retains hash; verifier understands redaction |
| Legal hold extends retention by editing entries | Editing breaks chain | Legal hold suspends destruction, doesn't edit entries |
| Destruction not recorded as audit event | Auditor cannot tell what was destroyed | audit.* destruction entries; permanent retention |
| Retention policy is uniform across decision_types | Some decisions are subject to longer holds than others | Per-decision_type policy |
| Subject-level keys destroyed without recording the destruction | Lost ability to prove destruction occurred | audit.key-destruction entry |
| Court-ordered destruction without waiver entry | Looks like tamper; no defence at later review | Legal-authority waiver with named order, signed by high-trust key |
| Maximum retention not addressed at all | EU GDPR storage-limitation principle violated; data accumulated indefinitely | Explicit max retention per decision_type or explicit case for indefiniteness |

## The Bottom Line

**Append-only and right-to-be-forgotten reconcile because the chain is over hashes, not plaintext. Cryptographic erasure (per-subject keys) destroys content while preserving chain. Redaction-with-witness handles legacy. Segregated-PII architecture prevents the conflict from arising in new builds. Legal-authority waivers handle the irreducibly hard cases. Every destruction is itself an audit event with permanent retention. Retention is a primary design property, not exception handling.**

---

**Retrieval test (run at end of build):** "GDPR Article 17 request arrives for subject S whose data appears in 412 entries across three decision_types. The pipeline uses cryptographic erasure with per-subject keys for one decision_type and segregated-PII for the others. Walk through the response."
