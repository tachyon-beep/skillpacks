---
name: using-experiment-formalisation
description: Use when formalising an experiment as a machine-checkable record layer using scientific-experiment ontologies — EXPO (Soldatova & King's ontology of scientific experiments), SUMO or BFO as upper ontology, PROV-O for lineage. Use when handed a mapping of your system onto EXPO and asked to check or extend it; when results must be auditable, reproducible, or interoperable with another group; when someone proposes making a knowledge graph the source of truth for decisions; when experiment records cannot say what a result was compared against, which regime produced it, or whether a value was unmeasured rather than zero. NOT for Expo/React Native, and NOT for Eclipse SUMO traffic simulation.
---

# Using Experiment Formalisation

## Overview

**Formalising an experiment means making its grammar machine-checkable: what was hypothesised, what varied, what it was compared against, what was measured, what was absent, who did what, and what may be concluded — as records a query can interrogate rather than prose a reader must trust.**

Two things go wrong, reliably, and this pack exists for both.

The first is **unverified vocabulary**, and it fails in both directions. Ask for a mapping of a system onto EXPO and you will get a confident table citing `Control`, `Replication`, `ResultSet`, `ExperimentRecord`, `Lineage`, `DataAnalysis`, `HypothesisTesting` — none of which are EXPO classes. Ask for a correction and you may be told `IndependentVariable` and `ExperimentalProtocol` are not either, which is also wrong: both ship. Meanwhile the terms people actually write — `Factor`, `AdminInfoAboutExperiment`, `QualityControlStrategy` — appear in the published paper and *never shipped in the ontology*, whose real fragments are `ExperimentalFactor`, `AdminInfoExperiment`, `QualityControl`, and (genuinely, in the source) `FalsePpositive`. A citation to a class that does not exist survives review because nobody checks, and collapses later, taking the correct rows with it.

The second is **inverted layering**. Formalisation produces something queryable and declarative, and the next suggestion is always that it become authoritative: retire the typed schemas, evaluate the decision gate with SHACL, let the knowledge graph be the single source of truth. This trades fail-closed enforcement for open-world description, in which a missing required property is not an error but an incomplete account.

This pack is governed by two laws that close both, and by a default posture — **Tier 1**, an annotation layer over records you already emit — that keeps the cost proportionate to the benefit.

## The Two Laws

> **Law 1 — Verified vocabulary only.** A term is presented as EXPO, SUMO, or PROV-O vocabulary only if it has been checked in a primary source. Every other term is declared as your own extension. *Verified*, *unverified*, and *refuted* are three different verdicts and must not be collapsed.

> **Law 2 — The projection law.** The ontology describes your typed contracts. It never becomes the runtime source of truth. On disagreement the contract wins and the ontology is corrected — enforced by a CI sync check, not by good intentions.

Both are stated with their tests in [`the-projection-law.md`](the-projection-law.md). Read that sheet first.

## When to Use

Use this pack when:

- You have been handed a mapping of your system onto EXPO (or any experiment ontology) and must check, correct, or extend it.
- Experiment records must be auditable — someone will ask which policy version, which evidence, which regime, which comparison.
- Results must be interoperable with another group, an archive, a journal, or a regulator.
- Your records cannot currently say what a result was compared against, whether a value was unmeasured rather than zero, or which supports were in place when it was produced.
- Someone is proposing that a knowledge graph become the authoritative record for decisions.
- You are deciding whether formalisation is worth its cost at all — this pack's most valuable output is sometimes "don't."

Do **not** use this pack when:

- The problem is enforcement, not description — telemetry defaulting to zero, tolerant readers, unversioned schemas. Load `/contract-engineering`.
- The problem is what may be concluded from paired branches — pairing, multiplicity, the winner's curse. Load `/counterfactual-statistics`.
- The problem is re-executing a past run exactly — seeds, snapshots, divergence. Load `/determinism-and-replay`.
- You mean **Expo** the React Native toolchain, or **Eclipse SUMO** the traffic simulator. Different things entirely.

## Start Here

1. [`the-projection-law.md`](the-projection-law.md) — the two laws, the sync check, and the red flags. Non-negotiable.
2. [`formalisation-triage.md`](formalisation-triage.md) — should you formalise at all; Tier 1 as default, the four conditions for Tier 2, the ROI heuristic, when to stop.
3. [`competency-questions-first.md`](competency-questions-first.md) — write the questions before naming a class; they are the acceptance tests.
4. [`expo-verified-inventory.md`](expo-verified-inventory.md) — what EXPO actually contains, its design rules, its extension points, the do-not-cite list.
5. [`mapping-a-system.md`](mapping-a-system.md) — the procedure: inventory → candidate → **verify** → fit verdict → gap register → artifacts.
6. Then the layer sheets your system needs, and [`validation-and-conformance.md`](validation-and-conformance.md) to make it testable.

## Sheet Index

| Sheet | Role |
|---|---|
| [`the-projection-law.md`](the-projection-law.md) | The two laws, the CI sync check, what to say when asked to invert the layering |
| [`formalisation-triage.md`](formalisation-triage.md) | Whether, how much, when to stop; Tier 1 default and Tier 2's four conditions |
| [`competency-questions-first.md`](competency-questions-first.md) | Questions before terms; positive and negative controls; the honest exit |
| [`expo-verified-inventory.md`](expo-verified-inventory.md) | EXPO's real inventory, design rules, Galilean/Baconian modes, do-not-cite list |
| [`sumo-upper-binding.md`](sumo-upper-binding.md) | What an upper ontology buys; category errors; SUO-KIF vs OWL lossiness; SUMO vs BFO |
| [`mapping-a-system.md`](mapping-a-system.md) | The mapping procedure and its six fit verdicts; the table is never the deliverable |
| [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md) | Factors, levels, the mandatory no-intervention arm, replication kinds, unit of analysis |
| [`measurement-uncertainty-and-absence.md`](measurement-uncertainty-and-absence.md) | Units, quantity kinds, uncertainty, and the four-state absence encoding |
| [`provenance-and-lineage.md`](provenance-and-lineage.md) | PROV-O reused properly; plan vs execution; retention of failures; RO-Crate |
| [`extending-without-forking.md`](extending-without-forking.md) | The mint ladder, IRI and version policy, modularity, the gap register |
| [`governance-role-extensions.md`](governance-role-extensions.md) | Role archetypes as relations; declared separations as queries; delegated authorisation |
| [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md) | States as attributes, transitions as events; influence coefficients; scaffolds and gates |
| [`validation-and-conformance.md`](validation-and-conformance.md) | SHACL vs OWL, the ten invariants, vacuity patterns, the CI gate |
| [`prior-art-map.md`](prior-art-map.md) | 30+ projects with verified maturity, and how to re-check when this map rots |
| [`adapting-this-pack.md`](adapting-this-pack.md) | What is load-bearing, what to swap, scaling down to one page and up to publication |
| [`expo-owl-inventory.json`](expo-owl-inventory.json) | **Authoritative** — all 324 EXPO classes, 78 properties, subclass axioms, labels and 125 disjointness entries, extracted mechanically from the shipped `expo.owl` |
| [`verified-terms.json`](verified-terms.json) | Curated layer — SUMO terms checked by hand, paper-vs-OWL corrections, the do-not-cite list. Loaded by the critic agent |

## The Failure-Mode Catalogue

The defects this pack exists to close. The `formalisation-critic` agent audits directly against this list.

1. **Hallucinated vocabulary** — class names attributed to EXPO or SUMO that do not exist there. Plausible, well-formed, uncheckable by the reader, fatal to the mapping's credibility. *(the-projection-law, expo-verified-inventory)*
2. **Verdict collapse** — reporting "not in EXPO" when the honest verdict is "I could not verify this," or citing an unverified term as though checked. Overclaiming in either direction. *(the-projection-law)*
3. **Secondary-source citation** — taking class names from the paper, a figure, a survey, or another mapping table instead of the shipped artifact. The published EXPO figure and prose disagree with the shipped OWL in 14 recorded places (`verified-terms.json` → `expo_corrections`); the artifact wins. *(expo-verified-inventory)*
4. **Mapping-table-as-deliverable** — a two-column table handed over as the formalisation. Not executable, not testable, not falsifiable. *(mapping-a-system)*
5. **Ontology as runtime source of truth** — SHACL or a reasoner on a decision path; typed schemas retired in favour of the graph; enforcement moved into open-world semantics. *(the-projection-law)*
6. **Formalisation debt** — the grammar changed, the ontology did not, nothing compares them. The projection law with no sync check. *(the-projection-law, validation-and-conformance)*
7. **Premature generalisation** — formalising a grammar still in motion, so every change costs a migration. Visible as a gap register that grows every sprint. *(formalisation-triage)*
8. **RDF cosplay** — triples emitted and never queried; no named consumer; "for FAIR" as the justification. *(formalisation-triage, competency-questions-first)*
9. **Reinvention over reuse** — a hand-rolled lineage or role vocabulary where PROV-O already has one. *(provenance-and-lineage, extending-without-forking)*
10. **Category errors** — a protocol modelled as a process, a role as a class, a mutable status as a subclass, a document conflated with its content. *(sumo-upper-binding)*
11. **Class/instance confusion** — modelling this specific run as ontology classes. The ontology grows when you run an experiment. *(expo-verified-inventory)*
12. **Disjointness violations** — asserting one entity as several classes the source ontology declares `owl:disjointWith`. EXPO's design strategies are pairwise disjoint, so a design needing a control arm *and* a graded strength *and* a staged schedule cannot be one strategy instance; assert it as one and a reasoner reports the ontology inconsistent. *(expo-verified-inventory, controls-counterfactuals-and-replication)*
13. **Unversioned or opaque IRIs** — no `owl:versionIRI`, no deprecation policy, environment or run identifiers baked into term IRIs, meaning silently redefined under a stable name. *(extending-without-forking)*
14. **Control not modelled** — no no-intervention arm; or one that exists but cannot be recorded as having won. *(controls-counterfactuals-and-replication)*
15. **Unit-of-analysis inflation** — paired branches from a common ancestor recorded as independent samples. *(controls-counterfactuals-and-replication)*
16. **Absence encoded as a value** — unmeasured becomes zero via a default, a `fillna`, a dropped null, or a `COALESCE` at read time. *(measurement-uncertainty-and-absence)*
17. **Provenance retrofitted** — lineage reconstructed from logs after the fact: plausible, unfalsifiable, wrong where it matters. *(provenance-and-lineage)*
18. **Failures not retained** — only successes annotated; null results and rejections dropped, deleting the denominator. *(provenance-and-lineage)*
19. **Unfalsifiable hypothesis records** — a "hypothesis" field holding a goal or a description; or mode drift, running exploratory and reporting hypothesis-driven. *(expo-verified-inventory)*
20. **Validation theatre** — shapes derived from the data they validate, fixtures generated by the producer under test, queries whose zero rows are called a pass, OWL relied on to reject anything. *(validation-and-conformance)*

## Boundary

This pack does **not** cover:

- **Contract enforcement** — fail-closed parsing, validity masks, authority-scoped writers, structural blinding. `/contract-engineering`. This pack describes what those records mean; that pack makes them impossible to violate.
- **Statistical inference** — pairing, multiplicity, abstention calibration, pre-registration. `/counterfactual-statistics`. This pack gets the comparison structure into the record; that pack says what may be concluded from it.
- **Deterministic re-execution** — seeds, snapshot completeness, divergence localisation. `/determinism-and-replay`.
- **Audit-trail cryptography** — hash chains, signed exports, immutable storage. `/audit-pipelines`.
- **Which subsystem boundaries should exist** — `/solution-architect`, `/system-architect`.
- **Ontology-based application development** — reasoners as inference engines in a product. This pack is about record semantics, and Law 2 keeps reasoning off the runtime path.

## Routing by Symptom

### "Someone handed us a mapping of our system onto EXPO — is it right?"

**Route to**: [`expo-verified-inventory.md`](expo-verified-inventory.md), then [`mapping-a-system.md`](mapping-a-system.md).

**Why**: assume it contains fabricated class names until each is checked — circulated tables reliably do. Verify every term against [`expo-owl-inventory.json`](expo-owl-inventory.json), which is extracted from the shipped OWL; **do not verify against the paper**, which disagrees with its own ontology. Mark each Verified / Unverified / Refuted, *then* assign a fit verdict. Correcting fit before checking existence wastes the effort on terms that do not exist — and beware the reverse error, since several terms commonly "corrected" as fake are real.

### "Leadership wants the knowledge graph to be the source of truth for decisions"

**Route to**: [`the-projection-law.md`](the-projection-law.md).

**Why**: the request is reasonable in its aims — declarative, auditable policy — and wrong in its mechanism. The sheet gives the three substitutions that deliver what was actually wanted (versioned policy records, artifacts generated from the contracts, a graph rebuilt from the event log) without putting open-world semantics on a decision path.

### "We want to publish/share our experiment records — where do we start?"

**Route to**: [`formalisation-triage.md`](formalisation-triage.md), then [`competency-questions-first.md`](competency-questions-first.md).

**Why**: name the consumer first. If there is one, its requirements determine the tier and often the vocabulary. If there is not, you are at risk of building triples nobody queries, and the triage sheet's honest exit applies.

### "Our results can't be compared / reviewers say the claim isn't supported"

**Route to**: [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md), then [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md).

**Why**: usually one of three structural omissions — no no-intervention arm, paired observations counted as independent, or results pooled across regimes and scaffold states that were never recorded. All three are record-layer defects, not analysis defects.

### "A downstream consumer read a missing measurement as zero"

**Route to**: [`measurement-uncertainty-and-absence.md`](measurement-uncertainty-and-absence.md), then `/contract-engineering`.

**Why**: this pack makes the record *say* absent — four states, with reasons, surviving serialisation. Only the contract layer makes the producer structurally unable to emit an unmarked absence. You need both halves, in that order of authority.

### "EXPO doesn't have a concept for X"

**Route to**: [`extending-without-forking.md`](extending-without-forking.md).

**Why**: often true — EXPO is a skeleton. Walk the mint ladder before minting, because roles, lineage, units, and absence all have prior art. Then mint in your own namespace, record the gap and the reason, and never assert equivalence to an IRI you cannot resolve.

### "We built the ontology — how do we know it's any good?"

**Route to**: [`validation-and-conformance.md`](validation-and-conformance.md).

**Why**: the question has a mechanical answer. Competency-question tests with positive controls, SHACL for validation (never OWL, which cannot reject anything), the ten zero-row invariants, and a sync check that fails loudly when the ontology stops describing the contracts.
