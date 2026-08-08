---
description: Formalise an experiment as a machine-checkable record layer — EXPO/SUMO/PROV-O with verified vocabulary only, the projection law (contracts win, ontology describes), Tier 1 by default, and a validation suite that cannot pass vacuously
---

# Experiment Formalisation Routing

**Ontology-layer pack: making an experiment's grammar machine-checkable — hypothesis, factors, comparison, absence, provenance, conclusions — without the ontology ever becoming the runtime source of truth. NOT for Expo/React Native, and NOT for Eclipse SUMO traffic simulation.**

Use the `using-experiment-formalisation` skill from the `axiom-experiment-formalisation` plugin to route to the right specialist sheet.

## The Two Laws

1. **Verified vocabulary only** — a term is cited as EXPO/SUMO/PROV-O only if checked in a primary source; everything else is declared as your extension. Verified, unverified, and refuted are three different verdicts.
2. **The projection law** — the ontology describes your typed contracts; it never becomes the runtime source of truth. Contracts win on disagreement, enforced by a CI sync check.

## Sheets

- **the-projection-law** - the two laws, the CI sync check, what to say when asked to invert the layering
- **formalisation-triage** - whether to formalise at all; Tier 1 default, Tier 2's four conditions, the honest exit
- **competency-questions-first** - questions before terms; positive and negative controls
- **expo-verified-inventory** - what EXPO actually contains; design rules; the do-not-cite list
- **sumo-upper-binding** - what an upper ontology buys; category errors; SUMO vs BFO
- **mapping-a-system** - the mapping procedure and its six fit verdicts; the table is never the deliverable
- **controls-counterfactuals-and-replication** - factors, levels, the mandatory no-intervention arm, unit of analysis
- **measurement-uncertainty-and-absence** - units, uncertainty, and the four-state absence encoding
- **provenance-and-lineage** - PROV-O reused properly; plan vs execution; retention of failures
- **extending-without-forking** - the mint ladder, IRI and version policy, the gap register
- **governance-role-extensions** - role archetypes as relations; declared separations as queries
- **lifecycle-and-staged-protocol-extensions** - states as attributes, transitions as events; scaffolds and gates
- **validation-and-conformance** - SHACL vs OWL, the ten invariants, vacuity patterns, the CI gate
- **prior-art-map** - 30+ projects with verified maturity
- **adapting-this-pack** - what is load-bearing, what to swap, scaling down and up

Plus two data artifacts: `expo-owl-inventory.json` (authoritative — all 324 EXPO classes, 78 properties, disjointness axioms) and `verified-terms.json` (curated corrections and the do-not-cite list).

## Commands

- `/formalise-experiment` - end-to-end staged orchestrator: triage → CQs → verified mapping → context/module → validation → sync check
- `/formalise-design` - the design layer alone: hypothesis mode, factors, comparison structure, unit of analysis
- `/map-to-expo` - produce or check a mapping with every term verified against the shipped OWL
- `/audit-formalisation` - adversarial audit against the failure catalogue via the formalisation-critic agent

## Agents

- `experiment-formalisation-architect` - producer SME; designs the formalisation, opinionated toward Tier 1 and toward "don't"
- `formalisation-critic` - critic SME; checks every cited term mechanically; refuses to rubber-stamp

## Cross-references

- Typed contracts the ontology describes → `axiom-contract-engineering`
- Statistics for the paired-branch experiments themselves → `yzmir-counterfactual-statistics`
- Tamper-evident decision history → `axiom-audit-pipelines`
- Whole-system determinism/replay → `axiom-determinism-and-replay`
