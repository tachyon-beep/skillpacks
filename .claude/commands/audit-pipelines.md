---
description: TDD-validated audit-grade decision pipelines - canonical encoding (RFC 8785 JCS), append-only decision logs, fingerprint chains, HMAC/Ed25519 signed exports, immutable storage, decision provenance, retention reconciled with right-to-be-forgotten, partial replay
---

# Audit Pipelines Routing

**Procedural decisions are first-class artifacts. Every decision has a verifiable provenance chain.**

Use the `using-audit-pipelines` skill from the `axiom-audit-pipelines` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-audit-pipelines/skills/using-audit-pipelines/SKILL.md` — this wrapper is a thin pointer.

## Sheets

- **decision-log-architecture** - what is a "decision," mandatory fields, where the log boundary sits
- **canonical-encoding-for-fingerprinting** - RFC 8785 JCS, gotchas (floats, map ordering, timezones, Unicode)
- **fingerprint-chains-and-integrity** - linked-hash and Merkle-style integrity, gap recovery, partial-trust verification
- **signing-and-export-integrity** - HMAC / Ed25519, key rotation, signature granularity, partial-export integrity
- **immutable-storage-patterns** - append-only, SQLCipher at rest, write-once-verify-many
- **decision-provenance** - linking outputs back to inputs + ruleset version + code version
- **audit-aware-logging-vs-observability** - what crosses into the audit trail vs ordinary observability
- **threat-model-for-audit-logs** - tamper, replay, time-shift, partial deletion, key compromise
- **retention-expiry-and-rtbf** - reconciling immutability with deletion obligations
- **partial-replay-from-trail** - reconstructing state from N entries
- **performance-budget-for-audit-grade-pipelines** - amortising vs streaming

## Commands

- `/scaffold-audit-trail` - canonical-JSON + chain + storage scaffolding
- `/verify-integrity` - replay a chain against signatures and fingerprints
- `/design-decision-log` - interactive elicitation of decision boundaries

## Agents

- `audit-architecture-reviewer` - reviews a system for decision points lacking provenance
- `integrity-auditor` - runs verification on a real trail and reports gaps

## Cross-references

- Replay infrastructure (state replay, not provenance replay) → `/determinism-and-replay`
- Static enforcement of audit-rules → `/static-analysis-engineering`
- Trust-boundary controls → `/security-architect`
