---
name: using-structure-synthesis
description: "Use when a generator emits typed graph/program candidates and needs legality, canonical identity, diversity or search-space checks."
---

# Structure Synthesis

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Generation contract

Declare the grammar, interface/shape rules, operator and resource ceilings, representation and candidate-identity semantics. Separate generation, legality verification and downstream utility evaluation when that separation is part of the system design. A search policy may legitimately use predicted utility; document its selection boundary and audit it rather than treating every guided search as invalid.

- Preserve representation round-trip fidelity and the grammar's operator/edge attributes. Topology-only isomorphism does not establish semantic equivalence.
- Apply cheap type/shape/acyclic checks before canonicalization; run the full gate on the canonical form. Reachability/dead-node treatment must agree with normalization rules.
- Canonicalize only under declared equivalences. Deleting a nonlinearity or overwriting a real edge during splicing changes semantics.
- Check idempotence and label/order invariance, including downstream-distinguished nodes and tied symmetric branches. Raw IDs must not silently decide canonical identity.
- Use an owned, versioned serialization/hash contract. Hash equality is not a proof of general program equivalence; define collision handling and the supported equivalence relation.
- Measure diversity after canonicalization and, where relevant, with functional probes. Distinct serializations can represent one candidate.
- Reverify mutations/recombinations and retain provenance of accepted, rejected and failed candidates. Selection bias and best-of-K claims need independent evaluation.
- Gate grammar expansion by rejection/verification cost and coverage evidence; larger search spaces are not automatically useful.

## Evidence and output

Produce a grammar/representation contract, verifier/canonicalizer specification or repair, with executable invariants and counterexamples. Report canonical/functional diversity, costs and unresolved equivalence limits as relevant. A clean audit may cite evidence ruling out the applicable failure patterns; finding a defect is never a quota.

## Fault-specific references

- Unsupported topology or cost growth: grammar/representation and search-space sheets.
- Invalid decoding: `validity-by-construction-vs-post-hoc.md`, `structural-verification.md`.
- Duplicate identity or semantic drift: canonicalization and semantic-hashing sheets together.
- Uniform pools: diversity, generation-strategy and learning-objective sheets.
- Archive mutation: `lineage-mutation-and-recombination.md`.
- Existing design audit: `synthesis-anti-patterns.md`; record checks that pass as well as confirmed findings.

Architecture-family selection belongs to neural architectures; embedding a candidate in a live growable network to dynamic architectures; requesting/admitting it via RL to morphogenetic RL; effect inference to counterfactual statistics.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [canonicalisation and normal forms](canonicalisation-and-normal-forms.md)
- [conditioning on context and contracts](conditioning-on-context-and-contracts.md)
- [diversity and mode collapse](diversity-and-mode-collapse.md)
- [equivalence detection and semantic hashing](equivalence-detection-and-semantic-hashing.md)
- [generation strategies](generation-strategies.md)
- [graph representations for generation](graph-representations-for-generation.md)
- [learning objectives for generators](learning-objectives-for-generators.md)
- [lineage mutation and recombination](lineage-mutation-and-recombination.md)
- [search space evolution and explosion control](search-space-evolution-and-explosion-control.md)
- [structural verification](structural-verification.md)
- [synthesis anti patterns](synthesis-anti-patterns.md)
- [typed graph grammars](typed-graph-grammars.md)
- [validity by construction vs post hoc](validity-by-construction-vs-post-hoc.md)
