---
name: using-experiment-formalisation
description: "Use when experiment records need ontology interoperability, verified EXPO/SUMO/BFO/PROV mappings, executable competency questions or auditable description. Do not activate for ordinary experiment execution."
---

# Experiment formalisation

Formalise only when a named consumer needs a question the current records cannot answer. Default to a bounded annotation layer over authoritative typed records.

## Work from the affected contract

1. Run formalisation triage: name the consumer and Must questions, existing answers, cost, owner and stop condition. An evidenced decision not to formalise is a successful outcome.
2. Inventory current records and draft queries before choosing terms. Verify vocabulary against versioned primary sources; distinguish verified, unverified, refuted and locally minted terms.
3. Map roles, factors, controls, pairing, measurement/absence and lineage without changing runtime truth. For this annotation workflow, typed contracts remain authoritative and a sync check detects drift.
4. Use Tier 1 unless stable grammar, a real query consumer, maintenance ownership and unmet Tier-1 questions justify graph/reasoner machinery.
5. Validate query positive/negative controls, shape/invariant applicability and synchronization with actual emitted records. Record source revisions, gaps and unsupported mappings.

## Scope and completion

Deliver competency questions, verified mapping, extension/gap register, necessary context/queries and check evidence. Do not add ontology machinery to solve a contract-enforcement or statistical-inference defect. Never convert an unresolved vocabulary claim into an authoritative ontology assertion.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Adapting This Pack for Your Own Needs | [adapting-this-pack.md](adapting-this-pack.md) |
| Competency Questions First | [competency-questions-first.md](competency-questions-first.md) |
| Controls, Counterfactuals, and Replication | [controls-counterfactuals-and-replication.md](controls-counterfactuals-and-replication.md) |
| EXPO: The Verified Inventory | [expo-verified-inventory.md](expo-verified-inventory.md) |
| Extending Without Forking | [extending-without-forking.md](extending-without-forking.md) |
| Formalisation Triage | [formalisation-triage.md](formalisation-triage.md) |
| Governance and Role Extensions | [governance-role-extensions.md](governance-role-extensions.md) |
| Lifecycle and Staged Protocol Extensions | [lifecycle-and-staged-protocol-extensions.md](lifecycle-and-staged-protocol-extensions.md) |
| Mapping a System | [mapping-a-system.md](mapping-a-system.md) |
| Measurement, Uncertainty, and Absence | [measurement-uncertainty-and-absence.md](measurement-uncertainty-and-absence.md) |
| Prior-Art Map | [prior-art-map.md](prior-art-map.md) |
| Provenance and Lineage | [provenance-and-lineage.md](provenance-and-lineage.md) |
| Binding to an Upper Ontology (SUMO) | [sumo-upper-binding.md](sumo-upper-binding.md) |
| The Projection Law | [the-projection-law.md](the-projection-law.md) |
| Validation and Conformance | [validation-and-conformance.md](validation-and-conformance.md) |

## Optional task entry points

- [audit-formalisation](../../commands/audit-formalisation.md): Audit formalisation for the affected experiment formalisation contract, with scoped source and verification evidence.
- [formalise-design](../../commands/formalise-design.md): Formalise design for the affected experiment formalisation contract, with scoped source and verification evidence.
- [formalise-experiment](../../commands/formalise-experiment.md): Formalise experiment for the affected experiment formalisation contract, with scoped source and verification evidence.
- [map-to-expo](../../commands/map-to-expo.md): Map to expo for the affected experiment formalisation contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [experiment-formalisation-architect](../../agents/experiment-formalisation-architect.md), [formalisation-critic](../../agents/formalisation-critic.md). No fixed reviewer count is required.
