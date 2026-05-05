# Immutable Storage Patterns

## Overview

**The chain proves what bytes were written. Storage decides whether they stay written, who can read them, and what happens to them at rest. A perfect chain on a mutable, attacker-readable database is not an audit trail; it is a leak with extra steps.**

This sheet specifies append-only storage models, encryption-at-rest, write-once-verify-many guarantees, the boundary between trail and registry, and the operational discipline (backup, restore, migration) that keeps storage from becoming the weak link. Retention policy, expiry, and right-to-be-forgotten are the next sheet (`retention-expiry-and-rtbf.md`); this sheet covers what stores the bytes and how.

## When to Use

Use this sheet when:

- Designing where audit entries live, in what format, with what access controls.
- Choosing between database, object store, append-only log, or hybrid storage.
- Defining backup, restore, migration, and disaster-recovery procedures for the trail.
- Producing the storage portion of `06-storage-and-retention.md`.

## Core Principle

> Append-only is a property the application enforces, the storage layer enforces, and the operational discipline enforces — three layers of agreement, no single layer trusted. A revoked DBA who can rewrite history breaks audit-grade storage; so does an application bug that updates rather than appends; so does an operations procedure that "rebuilds the table" during migration.

## What "Immutable" Means

Immutability is a spectrum, not a boolean. State the chosen point in `06-`.

| Level | Property | Typical implementation |
|-------|----------|------------------------|
| 1 — Append-only by convention | Application code never updates or deletes | Discipline; code review |
| 2 — Append-only by schema | Database schema lacks UPDATE/DELETE permissions for the application user | DB grants; application has INSERT only |
| 3 — Append-only by storage layer | Storage layer rejects updates and deletes outright | Object store with object lock, immutable bucket policies, append-only log products |
| 4 — WORM (write-once-read-many) | Hardware or firmware enforces immutability for a retention period | WORM-capable object stores (S3 Object Lock Compliance mode, Azure Blob Immutable Storage, GCS retention policy locked) |
| 5 — Air-gapped / external custody | Periodic exports to a custodian that is not the producer | Notary services, public ledger anchors, regulator-held copies |

A pipeline composes levels: tier S typically achieves Level 2; tier M+ should achieve Level 3 or higher; tier L+ regulated systems often require Level 4 with periodic Level 5 anchoring (matches the external time anchoring discussed in `fingerprint-chains-and-integrity.md`).

**The trap at Level 1:** developers add an "audit fix" UPDATE statement during an incident response and the trail loses integrity months later when the next audit notices an entry no longer matches its hash. Level 1 is *not* sufficient for any tier where the trail is evidence.

## Storage Model Choices

### Relational database with append-only schema

Entries written as rows; primary key is `entry_id`; INSERT-only; UPDATE/DELETE permissions revoked from the application user.

**Strengths:** familiar tooling, transactional INSERTs, indexable for query, available everywhere.

**Weaknesses:** DBAs typically retain UPDATE/DELETE privilege for emergencies; relational databases are designed for mutability and the immutability is layered on top; backup/restore tooling can silently rewrite history during operations.

**Use when:** internal pipeline at tier S, query patterns dominate access, the team already has SQL operational expertise.

**Discipline:** application user has INSERT-only; DBA actions are themselves audited (separately); test that UPDATE fails as part of CI.

### Append-only object store with object lock

Each entry (or batch) becomes an object; the bucket has retention policy and object-lock enabled (compliance mode for regulated tiers); deletes are blocked by the storage layer until retention expires.

**Strengths:** Level 3-4 immutability enforced by storage; deletion requires retention expiry, not application code; cheap at high volumes; cross-region replication is straightforward.

**Weaknesses:** queryability is poor without a separate index; small-object proliferation has cost implications; object-lock semantics differ subtly between cloud providers.

**Use when:** tier M+ where regulatory immutability is required, throughput is high, query patterns are time-range or specific-id (not aggregate analytics).

**Discipline:** lock retention enabled at bucket creation, not after; lock policy itself is locked (S3 Object Lock with a retention-period-cannot-be-shortened policy); credential separation between the application (write) and operator (read).

### Append-only log product

Kafka with a durable topic, AWS Kinesis with extended retention, Pravega, distributed transaction logs.

**Strengths:** native append-only; high throughput; fan-out to consumers; ordering is intrinsic.

**Weaknesses:** retention is finite (Kafka topic retention rolls); not natively WORM; query is consumption-style, not point-lookup.

**Use when:** the audit pipeline is part of an event-driven architecture; entries naturally fan out to multiple consumers (storage, indexing, analytics); coupling to existing log infrastructure is acceptable.

**Discipline:** the log is a buffer; durable storage is downstream (the consumer that writes to object store, ledger, or relational DB). Don't treat the log itself as the trail — its retention will outlast the audit obligation in some cases and under-last in others.

### Embedded encrypted database

SQLite-class single-file embedded databases, often paired with file-level encryption (e.g., SQLCipher) for clients that hold their own audit subset.

**Strengths:** self-contained, file-level portability, encryption-at-rest by default, suitable for client-side or device-side trails.

**Weaknesses:** not multi-writer; backup is file-snapshot; corruption recovery is manual.

**Use when:** the producer is a single process or a single device, the trail is exported to a central system periodically, and storage is an integrity layer rather than a queryable archive.

**Discipline:** file-level WORM at the destination once written and exported; the embedded copy is the producer's working set, not the canonical trail.

### Hybrid

Hot path: append-only database or log for low-latency writes. Cold path: object store with object lock for long-term retention. Index: separate read-model for query performance.

**Strengths:** matches access patterns; cost-optimised; preserves immutability where it matters most.

**Weaknesses:** three storage systems to keep consistent; "the trail" exists in three places, all must agree; migration between hot and cold is itself an operation that can break integrity.

**Use when:** tier M+ with high write throughput and a long retention obligation.

**Discipline:** hot-to-cold migration is its own audit event (a migration entry in the chain) and produces a verification step before the hot copy is purged.

## Encryption at Rest

Audit entries often contain decision-relevant data — sometimes PII, sometimes commercially sensitive, often legally privileged. Encryption-at-rest is mandatory at L+, recommended at M.

Two distinct goals, often confused:

1. **Confidentiality at rest.** Bytes on disk are encrypted; possession of the disk is insufficient to read. Achieved by full-disk encryption (LUKS, BitLocker), database-level transparent encryption (TDE), or file-level encryption (SQLCipher, gocryptfs). All are equivalent for this goal at the *physical media* level.
2. **Confidentiality from the storage operator.** Cloud provider, DBA, or operator cannot read the bytes. Achieved by application-side encryption with keys held outside the storage system — typically a KMS-issued data-encryption-key, with the storage operator never seeing plaintext.

Goal 1 is satisfied by transparent encryption; the operator can still read. Goal 2 requires application-side encryption.

State the goal in `06-`. "Encrypted at rest" is ambiguous; "encrypted with application-managed keys held outside the storage operator's control plane" is specific.

### Crypto-erase and retention

Encryption-at-rest with key-management is also the foundation for cryptographic erasure (see `retention-expiry-and-rtbf.md`). Retiring a per-record key destroys the record without rewriting storage. This makes encryption-at-rest a retention mechanism, not just a confidentiality mechanism.

## Boundary Between Trail and Registry

The trail (entries) and the registries (rulesets, inputs, code/SLSA) are *different stores* with different properties:

| Aspect | Trail | Ruleset registry | Inputs registry |
|--------|-------|------------------|-----------------|
| Append-only | Strict | Strict | Strict |
| Mutable metadata | None | Approval status flags ARE permissible (with their own audit entries) | None |
| Access control | Read-restricted | Often read-broader (developers may need to inspect rulesets) | Read-restricted (often contains PII) |
| Encryption-at-rest | Required at L+ | Recommended | Required at L+ |
| Operator separation | Yes | Should be separate from trail operator | Should be separate from trail operator |
| Retention obligation | Per regulator / contract | Aligned with trail (no orphan refs) | Aligned with trail |

**Single-store anti-pattern:** putting trail, ruleset, and inputs in the same database. One credential, one administrative interface, one attacker target. Separation of duties is structural, not cultural.

## Backup, Restore, Migration

### Backup

Audit storage is backed up like any other. Two specifics:

- **Backup is itself an audit event** for tier L+. A backup-completion entry in the chain commits to "the trail as of entry hash H is preserved at backup B."
- **Backups are themselves WORM** if the trail is. A backup that can be edited defeats the immutability of the source.

### Restore

Restoring from backup recovers entries; it does not recompute hashes. Restored entries verify against their *original* canonical bytes (which is why the bytes, not the database row format, are what the chain covers). A restore that produces hashes that don't match the original bytes means the backup format mutated the bytes — typically a serialiser or charset issue — and the restore has corrupted the trail.

Restore-as-incident: every restore is documented as an incident with a timeline, scope, and verification. A silent restore is indistinguishable from tampering.

### Migration

Storage migrations (database upgrade, object store change, hot-to-cold tier transition) preserve the bytes the chain covers, byte-for-byte. The migration produces a migration-event entry in the chain with:

- pre-migration head hash
- post-migration head hash (must equal pre-migration if no entries were added during migration)
- migration evidence reference (timestamps, operator identity, verification report)

A migration that *intends* to preserve bytes but does not (e.g., the new store normalises whitespace in JSON) breaks the chain; this is a chain-breaking event in `03-` terms and must be planned as one.

### Storage-level corruption

A bit flip on a storage medium, a partial write, or filesystem corruption can break individual entries' hashes without an attacker. Detection happens on verification (the entry doesn't hash to its stored `entry_hash`). Recovery: restore the affected entries from backup; if the affected entries are also corrupt in backup, treat as data loss and seal a gap-marker entry per `fingerprint-chains-and-integrity.md`. Storage corruption is a normal failure mode; the gap-marker discipline handles it cleanly when it happens.

## Operational Discipline

### Application user has INSERT-only

The application user that writes audit entries has *no* UPDATE, DELETE, ALTER permissions on audit tables. This is enforced by the database, not by the application code. CI tests verify this by attempting an UPDATE and confirming failure.

### Operator access is separately audited

Operators (DBAs, SREs, on-call) accessing audit storage do so through an interface that audits *the access*. Read access is logged; write access (which should be impossible) raises immediate alarm. Without operator-action auditing, the immutability of the trail extends only to what the application can do, not to what privileged humans can do.

### Schema migrations are blocked from the trail tables

Schema migrations on audit tables are extraordinarily dangerous (a column rename, a default-value change, a charset change can mutate stored bytes). Such migrations are blocked by policy except via a documented chain-breaking-event procedure. The schema-of-the-storage and the canonical-encoding-of-the-entry are decoupled deliberately: the storage-layer schema may evolve as long as the canonical bytes the chain covers remain byte-identical on read.

### Backup verification cadence

A backup never restored is a backup that doesn't exist. Schedule restore drills at a cadence proportional to the tier:

- Tier S: annually
- Tier M: quarterly
- Tier L+: monthly, with verification that restored entries hash correctly

Drill records are themselves audit entries.

## Spec Output (storage portion of `06-storage-and-retention.md`)

Answers, in order:

1. **Immutability level** — 1 to 5, per storage layer.
2. **Storage model** — relational, object-store, append-only log, embedded, hybrid, with rationale.
3. **Encryption-at-rest** — physical-media level, application-level, KMS, key-custody (cross-ref `04-`).
4. **Trail-vs-registry separation** — store identities, operator separation, access-control summary.
5. **Backup discipline** — cadence, location, immutability of backup itself, backup-event entries.
6. **Restore protocol** — verification step, incident framing, who can authorise.
7. **Migration protocol** — chain-breaking-event handling, byte-preservation verification.
8. **Operator audit** — what is logged, where it lives (must not be the same store as the trail).
9. **Drill cadence** — restore drills with their own entries.
10. **Cross-pack handoff** — `ordis-security-architect`'s controls covering storage (KMS, IAM, network); `axiom-devops-engineering`'s deployment of storage.

The retention/expiry/RTBF portion is in the next sheet — they share `06-` because they are tightly coupled, but the responsibilities are distinct.

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| Application user has UPDATE on audit tables "in case we need it" | Discipline becomes optional; one outage and history is rewritten | INSERT-only enforced by DB; tested in CI |
| WORM retention enabled but configurable post-hoc | Adversary disables WORM, edits, re-enables | WORM policy locked at bucket creation, policy-cannot-be-weakened |
| Backup is read-write S3 with no object lock | Backup mutability defeats source immutability | Backup destination is also WORM |
| "Encrypted at rest" means TDE only | Storage operator can read plaintext | Application-side encryption with KMS-managed keys for L+ |
| Trail, ruleset, inputs in one database | Single attack vector for all three | Separate stores, separate operator accounts |
| Restore is silent | A restore is indistinguishable from tampering | Restore-as-incident; documented timeline; verification |
| Storage migration that mutates byte form | Chain breaks silently; entries fail to verify months later | Pre/post-migration hash verification; chain-breaking event if bytes change |
| Long retention without restore drills | Backup is theatre | Drill cadence by tier; drill entries in the chain |
| DBA actions on audit storage not audited | Insider threat is unaddressed | Operator-action audit, in a separate store |
| Hot-to-cold migration without consistency proof | The hot copy is purged before cold is verified | Migration entry, verify cold, then purge |

## The Bottom Line

**Append-only at three layers (application, storage, operations); encryption-at-rest with application-side keys at L+; trail and registries in separate stores under separate operators; backups with the same immutability as the source; restores and migrations are documented audit events with their own chain entries; restore drills proportional to tier. Storage is where audit-grade pipelines collapse silently — do not let it.**

---

**Retrieval test (run at end of build):** "We're at tier M with a relational database storing entries and a separate object store for inputs. The DBA needs to run a schema migration to add an index. Walk through what is and isn't permitted."
