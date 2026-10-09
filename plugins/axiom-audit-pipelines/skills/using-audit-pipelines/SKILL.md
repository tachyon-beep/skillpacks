---
name: using-audit-pipelines
description: "Use when decisions require verifiable provenance, tamper-evident exports, or defensible retention and replay limits. Ordinary application logging does not require this workflow."
---

# Audit pipelines

Produce a decision-evidence contract that a separate verifier can check. Preserve the difference between an operational event, a recorded decision, and proof of that decision.

## Work from the affected contract

1. Identify the decision types, producers, consumers, adversaries and evidence obligation. Reuse the existing schema and storage when repairing one failure.
2. Bind inputs, ruleset/policy version, code identity and output. Specify the canonical bytes and version boundary before choosing hashes or signatures.
3. Choose chain/export construction, key custody, rotation and recovery against the threat model. State residual risks and independent verification steps.
4. Reconcile content retention/redaction with integrity witnesses and legal requirements verified for the deployment. State what the trail cannot reconstruct.
5. Exercise canonicalization vectors, alteration/deletion, key rotation, restore and bounded export cases applicable to the change. Record results and omissions.

## Scope and completion

A local repair needs the affected contract, evidence and checks. For a new audit system, use the scope tiers in decision-log-architecture.md and consistency checks in the references; create numbered artifacts only when their consumers need them. A signed chain proves integrity under stated custody assumptions, not that recorded inputs were truthful.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Audit-Aware Logging vs Observability | [audit-aware-logging-vs-observability.md](audit-aware-logging-vs-observability.md) |
| Canonical Encoding for Fingerprinting | [canonical-encoding-for-fingerprinting.md](canonical-encoding-for-fingerprinting.md) |
| Decision-Log Architecture | [decision-log-architecture.md](decision-log-architecture.md) |
| Decision Provenance | [decision-provenance.md](decision-provenance.md) |
| Fingerprint Chains and Integrity | [fingerprint-chains-and-integrity.md](fingerprint-chains-and-integrity.md) |
| Immutable Storage Patterns | [immutable-storage-patterns.md](immutable-storage-patterns.md) |
| Partial Replay from an Audit Trail | [partial-replay-from-trail.md](partial-replay-from-trail.md) |
| Performance Budget for Audit-Grade Pipelines | [performance-budget-for-audit-grade-pipelines.md](performance-budget-for-audit-grade-pipelines.md) |
| Retention, Expiry, and Right-to-be-Forgotten | [retention-expiry-and-rtbf.md](retention-expiry-and-rtbf.md) |
| Signing and Export Integrity | [signing-and-export-integrity.md](signing-and-export-integrity.md) |
| Threat Model for Audit Logs | [threat-model-for-audit-logs.md](threat-model-for-audit-logs.md) |

## Optional task entry points

- [design-decision-log](../../commands/design-decision-log.md): Design decision log for the affected audit pipelines contract, with scoped source and verification evidence.
- [scaffold-audit-trail](../../commands/scaffold-audit-trail.md): Scaffold audit trail for the affected audit pipelines contract, with scoped source and verification evidence.
- [verify-integrity](../../commands/verify-integrity.md): Verify integrity for the affected audit pipelines contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [audit-architecture-reviewer](../../agents/audit-architecture-reviewer.md), [integrity-auditor](../../agents/integrity-auditor.md). No fixed reviewer count is required.
