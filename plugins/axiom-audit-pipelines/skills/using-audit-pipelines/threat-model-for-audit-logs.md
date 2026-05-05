# Threat Model for Audit Logs

## Overview

**The audit log is the evidence. The threat model in `ordis-security-architect` covers the *system* that produces decisions. This sheet covers the *log itself* — adversaries who attack the evidence, the controls that defend it, and the residual risks an honest spec accepts.**

A perfect chain over a perfectly-canonicalised, perfectly-stored trail still has adversaries. They want to add entries that didn't happen, suppress entries that did, alter time so something appears earlier or later, deny later that the producer ever signed an entry, leak entries to people who shouldn't see them, or use the legal system to compel access that bypasses the technical controls.

This sheet enumerates the standard adversaries, names the standard controls, and tells you which ones honest tier S/M pipelines accept as residual risk.

## When to Use

Use this sheet when:

- Producing `07-threat-model.md`.
- Cross-checking that the design choices in `01-` through `06-` actually defend against the relevant threats.
- Cross-referencing `ordis-security-architect`'s system threat model rather than duplicating it.

## Core Principle

> The threat model is honest about what the design defends, what it accepts as residual risk, and what it requires from operations and from the legal/compliance environment. A threat model that says "all attacks are defended" is wishful; a threat model that names the residuals and the conditions under which they become real is evidence-grade.

## Cross-Reference With ordis-security-architect

The boundary:

- **System-level threats** — adversaries against the application, infrastructure, identity system, network, supply chain — are owned by `ordis-security-architect`. Threats at the producer layer (someone steals the producer's credentials, the producer's code is compromised, the producer's host is rooted) live there.
- **Audit-log threats** — adversaries against the *evidence* — are owned here. Tamper, replay, time-shift, partial deletion, key compromise of the *audit signing keys*, insider threats against the audit store, regulator subpoena.

The two overlap. A compromised producer can write any entries it wants — that is a system threat (control: minimise blast radius of a compromised producer through least-privilege, isolation, observation) *and* an audit threat (the entries are technically chain-valid; the residual risk is "we cannot tell if a chain-valid entry was written by a compromised producer"). The threat appears in both threat models with cross-references; the responses live in their respective controls.

`07-` lists the audit-log adversaries and references the system threat model where the system controls are the load-bearing element.

## Standard Adversaries

### Adversary 1 — Tamper (T)

**Goal:** Alter an existing entry's content (change `output`, change `inputs_commitment`, replace `ruleset_version`) so the trail says something different from what happened.

**Attack vector:**

- Direct database write (insider with UPDATE privilege; cloud account compromise; backup-restore cycle that mutates).
- Storage-layer mutation that bypasses application controls.
- Migration that "normalises" bytes and changes hashes.

**Controls:**

- Append-only at three layers (application, storage, operations) — `06-`.
- Chain integrity: an altered entry's hash no longer matches its stored `entry_hash` — `03-`.
- Per-entry signatures: the altered entry's signature doesn't verify — `04-`.
- External anchoring (notary, public ledger): a published anchor before the alteration prevents covering up.

**Residual risk:** A tamper that alters *all* affected entries' hashes, signatures, and the head — and any external anchors — is detectable only at scale (the alteration is large enough to reach anchors). Tier L+ residuals require external anchoring; tier S accepts the residual that an attacker with full producer + storage compromise can rewrite recent history before the next anchor.

### Adversary 2 — Replay (R)

**Goal:** Re-emit an old entry (or a copy of one) so the trail shows the same decision happening again, possibly in a different temporal context.

**Attack vector:**

- Re-injecting a known-good entry into the chain.
- Forking the chain and merging back later, with one branch containing replayed entries.

**Controls:**

- Each entry includes a unique `entry_id` (UUIDv7 / ULID). Duplicate `entry_id` is rejected by the application — `01-`.
- `prev_hash` linkage means a replayed entry has a wrong predecessor; chain construction rejects it — `03-`.
- For Merkle constructions, the inclusion proof would have to forge a tree consistency proof — much harder.
- Timestamp `decided_at` provides an additional consistency check; replays at a later wall-clock time are detectable if the rest of the chain advances.

**Residual risk:** A producer that replays its own entries (e.g., to inflate metrics) cannot be defended by the chain alone — the chain validates a producer-attested entry. Detection comes from external evidence (the recipient of the decision says "this happened twice"). System-level threat (cross-ref ordis).

### Adversary 3 — Time-Shift (T-shift)

**Goal:** Make an entry appear to have been produced at a different time. Often combined with tamper or replay.

**Attack vector:**

- Producer manipulates `decided_at` at write time (the producer chose its own clock).
- Producer rewrites `decided_at` after the fact (defeated by chain integrity, but the *original* assertion was already chosen).
- Verifier presented with a modified `decided_at` and no external time anchor.

**Controls:**

- External time anchoring: notary co-sign at intervals, RFC 3161 TSA tokens, public ledger commitments — `03-`.
- Producer clock-discipline: NTP/PTP, monitoring of clock skew (system control, cross-ref ordis).
- Cross-evidence: timestamps in observability are imperfect but exist; aligned external clocks (regulator's clock, recipient's receipt) bound time.

**Residual risk:** Any entry between two notary anchors has time only bounded to that interval. Producers without external anchoring at tier S are trusted on their own timestamps — explicit acceptance in `07-`.

### Adversary 4 — Partial Deletion (D)

**Goal:** Suppress specific entries that should be in the trail. The remaining trail looks complete.

**Attack vector:**

- Direct delete (defeated by append-only at storage layer).
- Truncate-and-rebuild (defeated by chain integrity if external anchors exist).
- Backup tampering: delete from production *and* from backup before the next anchor.

**Controls:**

- Append-only storage with WORM at L+ — `06-`.
- Chain construction means a removed entry leaves a gap that a verifier detects — `03-`.
- External anchoring: an anchor before the deletion bounds the deletion's possible scope.
- Gap-marker entries make detected gaps explicit — `03-`.

**Residual risk:** A deletion that occurs before any external anchor, of contiguous trailing entries, with backup compromised, leaves no fingerprint inside the local trail. External anchoring is the only defence; tier L+ requires it.

### Adversary 5 — Key Compromise (K)

**Goal:** Obtain a producer's signing key and forge entries.

**Attack vector:**

- Application memory disclosure.
- KMS misconfiguration.
- Operator with key access.
- Supply-chain compromise of the signing library.

**Controls:**

- Key custody at the strongest level the tier requires (HSM at L+) — `04-`.
- Key rotation cadence and triggers — `04-`.
- Compromise response (revocation entry, affected window, re-signing) — `04-`.
- Cryptographic agility (preemptive migration when a scheme weakens) — `04-`.
- Separation: signing key custody operator ≠ trail storage operator.

**Residual risk:** Forged entries between key compromise and revocation are chain-valid and signature-valid. Their signatures are by the compromised key, but the verifier doesn't know that until revocation is published. The "affected window" policy in `04-` defines how these are treated — typically downgraded to "suspect" until corroborated by external evidence.

### Adversary 6 — Insider (I)

**Goal:** Privileged human (DBA, SRE, ops, the producer's developer) bypasses the technical controls using their privilege.

**Attack vector:**

- DBA disables append-only enforcement, alters entries, re-enables.
- Operator restores from a maliciously prepared backup.
- Developer pushes a code change that disables audit emission for specific decisions.
- Compliance officer suppresses entries before an audit.

**Controls:**

- Operator-action audit (operator access logged in a separate store; cross-ref system threat model).
- Separation of duties: storage operator ≠ application developer ≠ key custody operator.
- WORM storage at L+ that *the storage operator cannot override*.
- Multi-party authorisation for high-impact actions (key rotation under compromise, schema migrations of audit tables).
- External anchoring (an insider at the producer cannot also forge external anchors).
- Code-change review with audit-affecting changes flagged (PR template, SAST rule).

**Residual risk:** A coordinated insider conspiracy across operator, developer, and key custody can defeat any internal control. The defences are organisational (background checks, role rotation, two-person rules) and external (anchoring with a third party). Tier L+ requires explicit insider-threat treatment in `07-`.

### Adversary 7 — Regulator Subpoena (S)

**Goal:** A legal order compels disclosure of trail entries — possibly more than the producer wants, possibly less than the recipient wants, possibly with a gag order preventing disclosure of the order itself.

**Attack vector:**

- Lawful but adversarial: the regulator requests data; the producer must comply; the requested scope may exceed contract or expectation.
- Lawful but secret: a gag order prevents notification.
- Conflicting jurisdictions: data subject to GDPR right-to-be-forgotten is requested by a non-EU regulator.

**Controls:**

- Documented disclosure runbook (legal/compliance owned, cross-ref `axiom-sdlc-engineering` governance).
- Encryption-at-rest with operator-blind keys: even a regulator request to the cloud provider does not yield plaintext (the cloud provider does not have the keys).
- Compartmentalisation: the trail contains only audit-relevant data; PII is segregated.
- Minimisation: the schema in `01-` rejects fields not needed for the audit purpose; less data, less scope of compelled disclosure.

**Residual risk:** A subpoena to the producer with the keys yields plaintext. The *legal* defences (challenging scope, narrowing the order, asserting privilege) are outside this pack. The *technical* posture (operator-blind keys, minimisation) bounds the worst case.

### Adversary 8 — Repudiation (Rep)

**Goal:** The producer later claims an entry isn't theirs ("we never decided that").

**Attack vector:**

- Symmetric (HMAC) signature schemes provide no defence — the verifier could have written it.
- A producer with poor key custody can plausibly deny ("our key was leaked, that entry isn't ours") — sometimes legitimately, sometimes adversarially.
- A producer that hasn't published their public key cannot have signatures verified by third parties.

**Controls:**

- Asymmetric signatures with public-key publication — `04-`.
- Clear key custody and rotation discipline — `04-`.
- Compromise response that distinguishes "before compromise" from "after compromise" entries — `04-`.

**Residual risk:** A producer who *legitimately* lost key control can repudiate the affected window. If repudiation is a frequent concern, tier should escalate (HSM custody, multi-party signing, external co-signing).

## STRIDE Cross-Reference

For readers used to Microsoft STRIDE:

| STRIDE | This sheet's adversary |
|--------|------------------------|
| Spoofing | A1 Tamper, A8 Repudiation (signing) |
| Tampering | A1 Tamper, A4 Partial Deletion |
| Repudiation | A8 Repudiation |
| Information disclosure | A7 Subpoena, system-level (cross-ref ordis) |
| Denial of service | Backpressure-on-audit (cross-ref `audit-aware-logging-vs-observability.md`); system-level (cross-ref ordis) |
| Elevation of privilege | A6 Insider, A5 Key Compromise |

A standalone STRIDE table is permissible at L+ where regulators expect the format. It cross-references the adversary list above; do not produce two unconnected tables.

## Tiered Threat Coverage

The required depth in `07-`:

| Tier | Required coverage |
|------|------------------|
| XS | Adversaries A1, A2, A4 (mentioned by name, controls cross-referenced; residual risks named) |
| S | A1-A4 plus A5; A6 and A7 named as out-of-scope with rationale |
| M | A1-A6; A7 named as out-of-scope with rationale |
| L | All eight, full controls map, explicit residual-risk register |
| XL | All eight, plus regulator-specific threats (jurisdictional conflicts, escalated subpoena scenarios), plus a STRIDE table |

## Spec Output (`07-threat-model.md`)

Answers, in order:

1. **Boundary statement.** This sheet covers the audit log; the system threat model lives in `ordis-security-architect`. Cross-references named.
2. **Adversary register.** Each tier-relevant adversary with goal, vector, controls (with cross-references to the producing sheet), and residual risk.
3. **Residual risk register.** Concise list of accepted residuals with the conditions under which they become real (e.g., "tier S accepts time-shift residual if no external anchoring is in place").
4. **STRIDE table** (L+).
5. **Compliance-driver mapping.** Which adversaries the relevant frameworks (SOC 2, HIPAA, PCI-DSS, GDPR, EU AI Act, NIS2) require explicit treatment of.
6. **Incident-response handoff.** Where in `axiom-sdlc-engineering` (or equivalent) the runbooks live for compromise events.
7. **Update cadence.** When the threat model is reviewed (annually at minimum; on any material design change; on any incident).

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| Threat model claims all attacks defended | Wishful; auditor distrusts wishfulness | Name residuals; explain conditions |
| STRIDE table at S tier | Format-driven, not threat-driven | Adversary register sized to tier; STRIDE only at L+ |
| No cross-reference to system threat model | Duplicate STRIDE rows; drift over time | Cross-ref ordis; do not duplicate |
| Insider threat named but not addressed | "We trust our DBAs" is not a control | Operator-action audit, separation of duties, WORM storage |
| Subpoena treated as a system threat | Different defences (legal + minimisation) | Separate adversary; runbook reference; minimisation-by-design |
| Key compromise response abstract | "Rotate the key" doesn't tell the on-call what to do | Runbook reference; affected-window policy from `04-` |
| HMAC chosen for cross-zone verification, repudiation listed as defended | HMAC ≠ non-repudiation | Asymmetric signatures or accept residual |
| Threat model never updated | Stale model fails audit | Annual review cadence; material-change triggers |

## The Bottom Line

**Eight standard adversaries (tamper, replay, time-shift, partial deletion, key compromise, insider, subpoena, repudiation), each with controls cross-referenced into the rest of the spec and explicit residual risks accepted at the chosen tier. The audit-log threat model cross-references the system threat model; it does not duplicate it. Honesty about residuals is the property auditors check.**

---

**Retrieval test (run at end of build):** "We're at tier M with HMAC signing internally and Ed25519 for exports, no external time anchoring. List the adversaries against which the design has accepted residual risk, and the conditions under which those residuals become real."
