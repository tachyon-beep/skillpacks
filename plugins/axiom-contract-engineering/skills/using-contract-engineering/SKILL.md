---
name: using-contract-engineering
description: "Use when typed records cross subsystem authority boundaries and absence, meaning changes, blinded views, or derived decisions must remain explicit and reproducible."
---

# Cross-boundary contracts

Make the producer/consumer agreement testable: what a record means, who can see it, how it evolves, and what fails when the agreement is violated.

## Work from the affected contract

1. Trace producers, readers and authority boundaries in current source. Identify the concrete mismatch or requirement before changing shared types.
2. Define fields, units, valid states and explicit absence. A measurement of zero is different from missing evidence; an intentional default needs declared semantics.
3. Version meaning changes and define supported producer/consumer combinations, unknown-version behavior, rollout and retirement. Strict audit/authority readers fail closed; select compatibility rules explicitly for other APIs.
4. Keep contract definitions independent of subsystem implementations. Derive records from recorded inputs; build blinded views by allowlist when information separation is required.
5. Test independent wire fixtures, missing/invalid/version cases, policy binding, resolver replay and forbidden-field canaries as applicable. Trace both sides of a change.

## Scope and completion

Deliver the changed contract, compatibility/authority decision, affected readers and verification evidence. A clean review is valid with inspected scope and explicit gaps. Do not invent a defect to satisfy a reviewer expectation.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Blinding by Construction | [blinding-by-construction.md](blinding-by-construction.md) |
| Canonical Identity | [canonical-identity.md](canonical-identity.md) |
| Contract-First Boundaries | [contract-first-boundaries.md](contract-first-boundaries.md) |
| Contract Testing | [contract-testing.md](contract-testing.md) |
| Definition Lifecycle | [definition-lifecycle.md](definition-lifecycle.md) |
| Dependency Direction | [dependency-direction.md](dependency-direction.md) |
| Deterministic Resolution | [deterministic-resolution.md](deterministic-resolution.md) |
| Schema Versioning and Evolution | [schema-versioning-and-evolution.md](schema-versioning-and-evolution.md) |
| Silent-Default Elimination | [silent-default-elimination.md](silent-default-elimination.md) |
| Versioned Policy Parameters | [versioned-policy-parameters.md](versioned-policy-parameters.md) |

## Optional task entry points

- [audit-contract-drift](../../commands/audit-contract-drift.md): Audit contract drift for the affected contract engineering contract, with scoped source and verification evidence.
- [design-contract-suite](../../commands/design-contract-suite.md): Design contract suite for the affected contract engineering contract, with scoped source and verification evidence.
- [review-contracts](../../commands/review-contracts.md): Review contracts for the affected contract engineering contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [contract-reviewer](../../agents/contract-reviewer.md), [contract-suite-architect](../../agents/contract-suite-architect.md). No fixed reviewer count is required.
