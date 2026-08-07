---
description: Typed cross-boundary contract discipline - schema design where illegal states are unrepresentable, versioning that fails closed, deterministic resolution, structural blinding, canonical identity, and contract testing
---

# Contract Engineering Routing

**Contract-layer pack: the records that cross subsystem boundaries and the discipline that keeps them honest. For whole-system replay use `/determinism-and-replay`; for REST/GraphQL surface design use `/web-backend`.**

Use the `using-contract-engineering` skill from the `axiom-contract-engineering` plugin to route to the right specialist sheet.

## Sheets

- **contract-first-boundaries** - typed immutable records, one authority per record class, illegal states unrepresentable
- **silent-default-elimination** - absent ≠ zero; validity masks; version-gated absence; no tolerant readers
- **schema-versioning-and-evolution** - meaning changes are new versions; fail-closed gates; expand/contract; no shims
- **deterministic-resolution** - pure resolvers over recorded inputs; canonical outputs; narrow-only authority
- **blinding-by-construction** - excluded fields absent from the view schema; allowlist projection; canary tests
- **canonical-identity** - content-addressed semantic hashing; cross-stage hash binding; raw-to-canonical traceability
- **dependency-direction** - contracts as leaf package; import-lint as a CI gate
- **versioned-policy-parameters** - thresholds/weights in versioned policy records; decisions bind the version in force
- **definition-lifecycle** - draft → approved → locked via recorded events; never silent edits
- **contract-testing** - golden wire fixtures, canonicalisation properties, schema-invalid rejection, authority tests

## Commands

- `/design-contract-suite` - forward-design a complete contract suite from a boundary inventory
- `/review-contracts` - audit a suite or diff against the 13-entry failure-mode catalogue
- `/audit-contract-drift` - cheap mechanical sweep for drift signals; CI-friendly

## Agents

- `contract-suite-architect` - producer SME; designs schemas, versioning, resolver, views, tests
- `contract-reviewer` - critic SME; severity-rated findings with evidence; refuses to rubber-stamp

## Cross-references

- Whole-system determinism/replay → `axiom-determinism-and-replay`
- Tamper-evident history over these records → `axiom-audit-pipelines`
- Which boundaries should exist → `axiom-solution-architect`
- HTTP/API surface above the records → `axiom-web-backend`
