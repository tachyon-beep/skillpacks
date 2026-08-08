# Prior-Art Map

**The purpose of this sheet is to stop you building what already exists — and to stop you depending on what is no longer maintained.** Both failures are expensive, and the second is the one nobody warns you about. A landscape map that marks only the live roads is worth less than one that marks the dead ends.

**Status column semantics** (all checked 2026-08-08 unless noted — see [Re-checking this map](#re-checking-this-map)):

| Status | Meaning |
|---|---|
| **Live** | Released or materially updated within roughly the last two years; you can file an issue and expect an answer. |
| **Stable** | Frozen on purpose. Not dead — finished. A W3C Recommendation is the archetype. |
| **Dormant** | The idea is published and citable, but the artifact is unmaintained, hard to obtain, or both. Read the paper; do not take a runtime dependency. |
| **Quiet** | Not dead, not moving. Last activity is old or cosmetic. Usable, but nobody is coming to fix it. |
| **Emerging** | Real and recent, not yet proven. Read it; do not build your foundation on it. |
| **Unverified** | Listed because you will encounter it; the maintenance status was not confirmed for this map. Check before relying on it — and note that six rows carrying this label in an earlier draft resolved to **Live** in minutes. An unverified label on a maintained project pushes readers away from good artifacts, which is the mirror of the failure this sheet warns about. |

---

## Upper ontologies

You need at most one, and possibly none (see [`formalisation-triage.md`](formalisation-triage.md)).

| Project | What it actually is | Status | Use when | Skip when |
|---|---|---|---|---|
| **SUMO** (Suggested Upper Merged Ontology) | Upper ontology in SUO-KIF (higher-order, LISP-like), plus MILO mid-level and domain files. Mapped to WordNet. Tooling: Sigma KEE. | **Live** — `ontologyportal/sumo` and `sigmakee` both have commits within days of this check (Aug 2026). | You are working with or extending EXPO (EXPO is built on SUMO), or you want a broad common-sense vocabulary with existing English glosses. | You need OWL-DL reasoning as the primary mechanism — SUMO's native expressivity is higher-order and the OWL translations are lossy. |
| **BFO** (Basic Formal Ontology) | Small, rigorously realist upper ontology. Continuant/occurrent split. `bfo-core.owl`. | **Live / Stable** — BFO-2020 is **ISO/IEC 21838-2:2021**; adopted by 650+ projects. | You are in or adjacent to biomedicine/life science, need an ISO-standard anchor, or intend to interoperate with any OBO Foundry ontology. | You want lightweight annotation only. BFO imposes real modelling discipline; that is its value and its cost. |
| **DOLCE** | Cognitively/linguistically oriented upper ontology (descriptive, not realist). DOLCE-Lite / DUL in OWL. | **Stable** — **ISO/IEC 21838-3:2023**, the same Top-Level Ontologies series as BFO's 21838-2, with normative OWL 2 and Common Logic axiomatisations. | You are modelling social, organisational, or role-heavy domains where descriptive stance beats realist stance. | You need community momentum — BFO and SUMO have more of it. |

**Choosing:** if you are extending EXPO, SUMO comes with it. If you must interoperate with life-science data, BFO. If neither pressure exists, you probably do not need an upper ontology at all — see [`sumo-upper-binding.md`](sumo-upper-binding.md) for what one actually buys you.

---

## Experiment and investigation models

| Project | What it actually is | Status | Use when | Skip when |
|---|---|---|---|---|
| **EXPO** | The generic ontology of scientific experiments: goals, design, model, planned actions, hypotheses, results, typed errors, admin metadata. **324 classes, 78 object properties** in the shipped `expo.owl`; OWL exported from Hozo. | **Dormant but obtainable** — 2006 paper is open access; the OWL exists and parses. SourceForge's file archive returns 403 to automated clients (hence the widespread claim that it is unavailable); retrieve through a browser. Unmaintained: at least five class names are misspelled in the shipped file. | You want the *grammar* of an experiment — design strategies, factors, target variables, hypothesis families, a typed error taxonomy — independent of domain. Uniquely good at this. | You expect it to be maintained, or you plan to `owl:imports` from a stable IRI. Re-express in your own namespace; cite the paper for rationale and the OWL fragments for names — see [`expo-verified-inventory.md`](expo-verified-inventory.md). |
| **OBI** (Ontology for Biomedical Investigations) | Large, actively maintained, BFO-anchored ontology of investigations: assays, devices, materials, study designs, roles. | **Live** — OBO Foundry member. | Your domain is life science, or you need a maintained, populated investigation vocabulary rather than a skeleton. OBI is what EXPO would have become had it been sustained. | Your experiments are computational and non-biological — you will be modelling around biomedical assumptions. |
| **ISA** (Investigation / Study / Assay) | A metadata *framework* and file formats (ISA-Tab, ISA-JSON), not an upper ontology. Three-level containment model. | **Live** — `ISA-tools/isa-api` active (commit 2026-07-23). | You need a pragmatic, tabular, tool-supported way to organise multi-study programmes. The I/S/A containment is a genuinely good default shape. | You need axioms and reasoning — ISA is structure and vocabulary references, not logic. |
| **CDISC** (SDTM, CDASH, ODM, Define-XML) | Clinical-trial data standards, not an ontology — the regulated world's answer to study data structure. Maintained by the CDISC consortium. | **Live** | Your experiments are clinical trials, or anything a regulator will read. | Non-regulated research; the overhead is real. |
| **LABORS** | Ontology developed for the Robot Scientist ("Adam"/"Eve") work; hierarchical experiment records for fully automated science. | **Dormant** — research artifact; cite the papers. | You want prior art for machine-generated experiments and deep experiment nesting — closest published relative to an autonomous experimentation loop. | You need a maintained artifact. |
| **AnIML ontology** | OWL 2 formalisation of AnIML — a long-established XML standard for analytical chemistry and biology data, widely used in industrial labs — aligned to the Allotrope Data Format. **The ontology is the new part, not the base standard.** | **Emerging** — CAiSE 2026 paper; artifact at `KE-UniLiv/animl-ontology`. | You are integrating instrument data across labs and want current work rather than settled work. | You need something proven. It is new. |

---

## Protocols, actions, and statistics

| Project | What it actually is | Status | Use when | Skip when |
|---|---|---|---|---|
| **EXACT2** | Ontology of *experimental actions* — the semantics of biomedical protocols, aimed at reproducibility. Finer-grained than EXPO's `PlanOfExperimentalActions` / `ExperimentalProtocol`. | **Dormant** — 2014 paper (PMC4255744); cite it. | You need to formalise a protocol at the level of individual executable steps. | You only need protocol identity and version, not step semantics. |
| **STATO** | General-purpose statistics ontology: tests, their conditions of application, distributions, variables, spread and variation metrics. Grew out of the ISA community. | **Live** — OBO Foundry listed; `ISA-tools/stato` active (commit 2026-04-20). | You are typing statistical claims — which test, under which assumptions, producing which estimate. EXPO deliberately does not go here. | Your statistics are simple enough that a versioned analysis-plan record covers you. |

**Statistical *method* is not statistical *validity*.** STATO lets you say which test you ran; it does not tell you whether the test was appropriate for paired branches sharing a common ancestor. For that, load `/counterfactual-statistics`.

---

## Provenance, workflow, and packaging

This layer is more mature than the experiment layer. Prefer it — do not reinvent lineage.

| Project | What it actually is | Status | Use when | Skip when |
|---|---|---|---|---|
| **PROV-O** | W3C provenance ontology. `Entity` / `Activity` / `Agent`, with `wasGeneratedBy`, `used`, `wasDerivedFrom`, `wasAttributedTo`, `wasInformedBy`. | **Stable** — W3C Recommendation. | Always, if you are recording lineage at all. This is the default answer and the most reusable thing in this map. | Never skip it in favour of a hand-rolled lineage vocabulary. That is the reinvention failure mode. |
| **P-Plan** | Extends PROV-O to separate the *plan* from its *execution* — plan steps and variables vs. the activities that realised them. | **Dormant** but small and stable enough to copy the pattern. | You need plan-vs-execution separation, which almost every experiment needs (the protocol you intended vs. the run you got). | You can express the same distinction with two record types and a `realization`-style link. Often you can. |
| **OPMW** | Workflow-level provenance, built over PROV/OPM; workflow templates and executions. | **Dormant** | You are formalising a workflow system specifically. | A general PROV-O activity graph suffices. |
| **RO-Crate** | Packaging: a directory plus a JSON-LD manifest describing what is inside and how it relates. Not an ontology — a *container* convention, mostly schema.org terms. | **Live** — 1.2 released June 2025, declared stable with minor fixes expected; RO-Crate 2 (modularisation) in progress. | You need to hand someone a self-describing bundle of an experiment. Strong choice for archival and exchange. | You need query over a knowledge graph rather than transport of a package. |
| **REPRODUCE-ME** | Ontology extending PROV-O *and* P-Plan specifically for scientific experiment provenance — sits exactly at this pack's intersection. | **Unverified** | You want prior art that already joined the experiment and provenance layers. | You have only one of the two problems. |
| **Workflow Run RO-Crate** | Published RO-Crate profiles joining packaging with run provenance — the bridge between this table's two recommended layers. | **Unverified** | You are packaging executed workflow runs, not just data. | Static datasets. |
| **Nanopublications** | Small, individually citable, attributed assertion units. | **Live** — active ecosystem (nanodash, Knowledge Pixels). | You need fine-grained citable claims with attribution. | You need whole-experiment records. |
| **DCAT / schema.org `Dataset`** | Dataset discovery and catalogue metadata. | **Stable** (DCAT is a W3C Recommendation; confirm which version you target) | You need your outputs to be *findable* by generic harvesters. | You need experimental semantics — these describe datasets, not experiments. |

---

## ML-specific

Mostly a graveyard with two live exceptions. Read this row-by-row before adopting anything here.

| Project | What it actually is | Status | Use when | Skip when |
|---|---|---|---|---|
| **Croissant** | MLCommons metadata format for ML-ready datasets: splits, label assignment, responsible-AI fields. JSON-LD over schema.org. | **Live** — 1.1 current; industry+academic backing. | Your artifacts are ML datasets and you want real tool support. Best-supported thing in this section. | You are describing the experiment, not the dataset. Croissant is dataset-scoped. |
| **ML-Schema** | W3C Community Group top-level schema for ML algorithms, datasets, models, experiments; designed as a mapping target for OntoDM, DMOP, Exposé, MEX. | **Dormant** — CG output, still cited in recent literature but not evidently maintained. | You want a small vocabulary skeleton for run/model/dataset and are willing to own it yourself. | You need maintenance or tooling. |
| **MEX vocabulary** | Deliberately lightweight ML experiment metadata, reusing PROV-O. | **Dormant** | You want a worked example of "small vocabulary layered on PROV-O" — the shape is instructive even if you do not adopt it. | You need it maintained. |
| **DMOP** (Data Mining OPtimization) | Data-mining workflow and meta-mining ontology. | **Dormant** — reported as not publicly available. | Reading the papers for meta-learning modelling ideas. | Anything operational. You likely cannot get the artifact. |
| **OntoDM / Exposé** | Data-mining and ML-experiment ontologies from the same research lineage. | **Unverified** | Literature review. | Production. |

**The pattern in this section is the lesson.** The ML community has produced many experiment ontologies and sustained almost none of them. The two survivors (Croissant, and PROV-O from the general web stack) survived by being *narrow and useful* rather than complete. Weigh that before you set out to build the general one.

---

## Units and measurement

| Project | What it actually is | Status | Use when |
|---|---|---|---|
| **QUDT** | Quantities, units, dimensions, data types as an ontology. | **Live** — release 2026-08-05. | You need units as first-class typed things in a graph. |
| **UCUM** | A code *syntax* for units, not an ontology. Compact, ubiquitous in clinical and lab data. | **Stable** | You need units as strings that machines can parse and convert. Usually enough. |
| **OM** (Ontology of units of Measure) | Units ontology, richer coverage of quantity kinds. | **Quiet** — last repository activity 2025-03-26 (a README edit). | QUDT does not cover your quantity kinds. |

**Default:** UCUM codes in a typed field. Escalate to QUDT/OM only when you must reason over dimensions. See [`measurement-uncertainty-and-absence.md`](measurement-uncertainty-and-absence.md).

---

## Cross-cutting vocabularies and principles

| Project | What it is | Status | Note |
|---|---|---|---|
| **SIO** (Semanticscience Integrated Ontology) | Broad general-purpose vocabulary for objects, processes, attributes in science. | **Live** — v1.59, commit 2026-05-01. | Useful gap-filler when neither EXPO nor your domain ontology has a term and you would rather reuse than mint. |
| **EDAM** | Bioinformatics operations, data types, formats, topics. | **Live** — release 1.25, 2026-06-26. | Domain-specific; only if you are in bioinformatics. |
| **OBO Foundry principles** | Not an ontology — a set of governance rules (open, common format, unique IRI space, versioning, documented, maintained, clearly bounded). | **Live** | **Read these even if you never touch an OBO ontology.** They are the best short statement of what makes a domain extension survivable, and they map directly onto [`extending-without-forking.md`](extending-without-forking.md). |
| **FAIR principles** | Findable, Accessible, Interoperable, Reusable. | **Live** | The "why" behind most of this map. Also frequently invoked to justify formalisation that has no consumer — see the RDF-cosplay failure mode. |

---

## Choosing quickly

Most projects need three things, not fifteen:

1. **A grammar for the experiment** → EXPO's concepts, re-expressed in your namespace ([`expo-verified-inventory.md`](expo-verified-inventory.md)).
2. **A lineage vocabulary** → PROV-O, reused directly, not reinvented ([`provenance-and-lineage.md`](provenance-and-lineage.md)).
3. **A packaging convention** → RO-Crate if anything leaves the building; otherwise nothing.

Add an upper ontology only if an external commitment demands it. Add STATO, EXACT2, QUDT, or a domain ontology only when a competency question you actually have cannot be answered without it.

---

## Re-checking this map

**This sheet will rot, and a rotted status column is worse than no status column** — it converts "I checked once" into "this is maintained." Before you rely on any **Dormant** or **Unverified** row, spend the two minutes:

- **Artifact reachable?** Fetch the ontology file at its documented IRI. A 404, a parked page, or a login wall is your answer.
- **Last real change?** Latest release/tag date on the repository, not the last commit to a README.
- **Anyone home?** Open issues with maintainer replies in the last year.
- **Still cited *as used*?** Recent papers *using* it, not merely listing it in a related-work table. This map's ML section is what "cited but unused" looks like.

Record what you checked and when, in the same place you record the mapping. A term inventory with a verification date is an asset; one without is a liability.
