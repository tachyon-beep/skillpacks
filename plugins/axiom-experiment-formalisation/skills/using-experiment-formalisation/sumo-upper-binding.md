# Binding to an Upper Ontology (SUMO)

**An upper ontology is a set of pre-made decisions about what kinds of thing exist.** You adopt one to stop re-litigating "is a protocol a thing or an event?" in every design meeting, and to make your terms mean something to people outside your project. You decline one when neither of those is a real problem — which is more often than ontology enthusiasm suggests.

This sheet is about SUMO because **EXPO is built on SUMO**: per the 2006 paper, 46 of the initial version's 218 concepts are reused SUMO classes (the shipped OWL declares 324 classes, so those figures describe an earlier state). If you re-express EXPO, you inherit that decision unless you deliberately replace it.

## What binding actually buys you

| Benefit | Real? | Condition |
|---|---|---|
| **Category discipline** — you cannot quietly model an event as an object | **Yes, always.** This is the main payoff and it accrues even if nobody else ever reads your ontology. | None. |
| **Shared meaning across projects** | Only if the other project binds to the same upper ontology *and* someone maintains the alignment. | A named external consumer. |
| **Reasoning / inference** | Rarely, in practice. Most projects never run a reasoner over their instance data. | You have a competency question that genuinely requires entailment. |
| **Reviewer credibility** | Yes, in academic contexts. Anchoring to SUMO or BFO is legible to reviewers. | You are publishing. |

If only the first row applies to you, **you can get it by reading this sheet and applying the discipline, without importing anything.** That is a legitimate outcome — see [`formalisation-triage.md`](formalisation-triage.md).

## SUMO in one screen

SUO-KIF, higher-order, LISP-like syntax. Actively maintained at [`ontologyportal/sumo`](https://github.com/ontologyportal/sumo); core in `Merge.kif`, mid-level in `Mid-level-ontology.kif` (MILO), plus domain files. Tooling: Sigma KEE.

The spine, verified in `Merge.kif`. **Some edges are elided** — `→` marks a path, not always a direct parent; intermediate classes are noted where they matter:

```
Entity
├── Physical            "An entity that has a location in space-time."
│   ├── Object
│   │   ├── AutonomousAgent → SentientAgent → CognitiveAgent
│   │   ├── ArtificialIntelligenceAgent  (direct subclass of AutonomousAgent)
│   │   └── … → SelfConnectedObject → CorpuscularObject → ContentBearingObject
│   │         "Any SelfConnectedObject that expresses content."
│   │         (also a subclass of ContentBearingPhysical — one of SUMO's
│   │          multiple-inheritance cases)
│   └── Process         "The class of things that happen and have temporal parts or stages."
│       └── IntentionalPsychologicalProcess
│           └── Investigating
│               └── Experimenting
└── Abstract            "Properties or qualities as distinguished from any particular embodiment."
    ├── Quantity → FiniteQuantity → PhysicalQuantity → {ConstantQuantity, UnitOfMeasure}
    ├── Attribute
    └── Proposition
        ├── Argument
        └── Procedure   "A sequence-dependent specification."
            └── Plan    "A specification of a sequence of Processes ... at some future time."
```

`Experimenting` is documented as: *"Investigating the truth of a Proposition by constructing and observing a trial. Note that the trial may be either controlled or uncontrolled, blind or not blind."*

### The relations you will actually use

CaseRoles: `agent` ("an active determinant of the Process"), `patient`, `instrument`, and `result` — whose documentation reads `(result ?ACTION ?OUTPUT)` means the output is a product of the action. (Do not write "a product of `Action`" as a quotation: SUMO has no class named `Action`, and inventing a capitalised term inside quotation marks is the same defect as citing a class that does not exist.)

Others: `subProcess`, `before`, `holdsDuring`, `hasPurpose`, `realization`, `containsInformation`, `represents`, `refers`, `measure`, `confersNorm`.

**`realization` is the one to notice** — but quote its actual signature, not a convenient reading of it: `(realization ?PROCESS ?PROP)`, domain 1 `Process`, domain 2 `Proposition`, a subrelation of `represents`, documented as a Process *expressing the content of* a Proposition (its examples are a musical performance of a score, a reading of a poem). Since a `Plan` is a `Procedure` and therefore a `Proposition`, this gives you plan-versus-execution for free. Note the argument order — process first — and that the semantics are representational rather than causal; "the process that carries out the plan" is your gloss, not SUMO's wording.

## Two verified facts that will save you an embarrassment

**1. `Agent` is not a SUMO class.** The class is `AutonomousAgent` (a subclass of `Object`); lowercase `agent` is a `CaseRole` predicate. Writing `sumo:Agent` is a citation to nothing. Note also `ArtificialIntelligenceAgent` exists as a sibling under `AutonomousAgent` — often what you actually want for an automated pipeline stage.

**2. There is no `Hypothesis` class in `Merge.kif` or MILO.** Zero occurrences in either file as of 2026-08-08. Hypotheses come from **EXPO** (`ExperimentalHypothesis`, `ResearchHypothesis`, `NullHypothesis`), not from SUMO. (SUMO ships further domain files that were not searched — so "not found where you would expect it," not "provably absent from all of SUMO." Either way, do not cite it without checking the specific file.)

Both are exactly the shape of error Law 1 exists to prevent: plausible name, correct-looking namespace, no referent.

## Category errors — the failure this sheet prevents

Every one of these is common, and every one silently breaks queries later.

| Error | Why it happens | The SUMO answer |
|---|---|---|
| **Modelling a protocol as a `Process`** | You "run" a protocol, so it feels like an event. | A protocol is a `Plan` — a `Procedure`, hence a `Proposition`, hence **`Abstract`**. The *run* is the `Process`. Link them with `realization`. |
| **Modelling a role as a class** | "Verifier" feels like a kind of thing. | A role is context-dependent — the same component may verify in one experiment and be verified in another. EXPO added a `Role` predicate on top of SUMO for exactly this. Model roles as relations, not subclasses. See [`governance-role-extensions.md`](governance-role-extensions.md). |
| **Modelling an attribute as a subclass** | `FailedRun` as a subclass of `Run`. | Status is an `Attribute`, and it changes. A subclass cannot change. Reify status as an attribute plus a transition event. |
| **Modelling a document as its content** | The record *is* the decision, surely. | `ContentBearingObject` (`Physical`) `containsInformation` a `Proposition` (`Abstract`). The file and the claim are different entities — and only one of them has a hash. |
| **Modelling a measurement as a number** | It is a number in your database. | A `PhysicalQuantity` carries a `UnitOfMeasure`. A bare float is a category error waiting to be a unit bug. See [`measurement-uncertainty-and-absence.md`](measurement-uncertainty-and-absence.md). |

**Cheap test:** ask *"does this thing have temporal parts?"* If yes it is a `Process`; if it can be written down and is true or false, it is a `Proposition`; if you can point at it in space, it is an `Object`; if it is how something is rather than what it is, it is an `Attribute`. Apply this once per term in your mapping table and most modelling arguments evaporate.

## A worked constraint: MILO's `experimentalControl`

MILO defines a control predicate:

```lisp
(instance experimentalControl CaseRole)
(subrelation experimentalControl patient)
(domain experimentalControl 1 Experimenting)
(domain experimentalControl 2 Object)
```

Read the second `domain` line. **Argument 2 must be an `Object` — something physical.** In a wet lab (a control sample, a control plate) that is correct. For a computational experiment whose control is a *no-intervention branch* — a process, or the plan describing one — `experimentalControl` does not fit its own domain constraint.

**This is what a real fit verdict looks like**, and it is why "SUMO has a control predicate, use that" is not an answer. Your options:

1. **Narrow your reading** — if your control genuinely is an object (a frozen checkpoint artifact, a fixed reference dataset), use it as-is.
2. **Mint a sibling** in your namespace with a domain that fits processes, and document the relationship to `experimentalControl` in prose.
3. **Model the control arm as a first-class entity** with its own identity, and relate it via your own predicate.

Option 2 or 3 for most computational work. Record it in the gap register either way — see [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md).

## SUO-KIF versus OWL — the practical problem

SUMO's native language is higher-order; OWL-DL is not. **The OWL translations of SUMO are therefore lossy**, and the axioms you were hoping to inherit are frequently among the losses. EXPO itself was authored in Hozo and auto-translated to OWL-DL.

Consequences:

- **Do not promise reasoning you cannot deliver.** "We inherit SUMO's axioms" is usually false in an OWL pipeline; you inherit the class hierarchy and the glosses.
- **Decide which artifact is authoritative** — the KIF or your OWL — and say so. Two authorities is the dual-source-of-truth failure.
- **If you need entailment**, write your own OWL axioms for the entailments you need and test them ([`validation-and-conformance.md`](validation-and-conformance.md)). Do not assume they arrived with the import.
- **In practice most projects need `rdfs:subClassOf` and clean glosses**, which survive translation intact. That is fine — just say that is what you are doing.

## When to use BFO instead

| Choose SUMO | Choose BFO | Choose neither |
|---|---|---|
| You are extending EXPO (which is SUMO-anchored). | You must interoperate with OBI, any OBO Foundry ontology, or life-science data. | Your formalisation is internal, has no external consumer, and you can get category discipline by reading this sheet. |
| You want broad common-sense coverage and English glosses. | You need an ISO-standard anchor (BFO 2020 is ISO/IEC 21838-2:2021). | Tier 1 in [`formalisation-triage.md`](formalisation-triage.md) — most projects. |
| You are comfortable with SUO-KIF, or need only the hierarchy. | You want strict continuant/occurrent discipline and a large maintained ecosystem. | |

**Do not bind to two.** Cross-walking SUMO and BFO is a research project, not a task. Pick one, record the choice and its reason, and note in the mapping that a future migration would be a major version — see [`extending-without-forking.md`](extending-without-forking.md).

## Binding checklist

- [ ] Decided whether you need an upper ontology at all, and recorded **why**.
- [ ] Chose exactly one; recorded which file and which retrieval date.
- [ ] Every upper-ontology term you cite is verified in that file and recorded in [`verified-terms.json`](verified-terms.json).
- [ ] Ran the four-way category test on every term in your mapping table.
- [ ] Plan-versus-execution is explicit (`Plan` + `realization` + `Process`, or the equivalent).
- [ ] Stated whether reasoning is actually used; if not, said so plainly rather than implying it.
- [ ] Named the single authoritative artifact.
