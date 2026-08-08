# Controls, Counterfactuals, and Replication

**A formalisation that cannot express what the result was compared against has formalised nothing.** This is the sheet people skip, and it is the one that determines whether the record can support a causal claim later.

EXPO gives you real vocabulary here — more than most people realise — and then stops short in two specific places. Both gaps are legitimate mints.

## What EXPO actually provides

Verified against the shipped OWL, not the paper's figure (see [`expo-verified-inventory.md`](expo-verified-inventory.md)):

| Concept you need | EXPO fragment | Notes |
|---|---|---|
| The thing you vary | `ExperimentalFactor` < `Variable` | **Not `Factor`** — that name never shipped. |
| Its settings | `FactorLevel` < `ValueOfVariable` | Including the "no intervention" level — see below. |
| "This variable is independent" | `IndependentVariable` < `Independence` < `AttributeOfVariable` | An **attribute attached to** a variable, not a kind of variable. Disjoint with `Controllability`. |
| The measured outcome | `TargetVariable` < `Variable` | `DependentVariable` also exists, as an attribute. |
| How many things vary | `One-factorExperiment` / `Two-factorExperiment` / `Multi-factorExperiment` | |
| The comparison scheme | `ExperimentalDesignStrategy` | Parent of the comparison subtree. |
| Paired comparison | `PairedComparison`, `PairedComparisonOfMatchingGroups`, `PairedComparisonOfSingleSampleGroups` | |
| Control vs. target groups | `ComparisonControl_TargetGroups` | Fragment uses an underscore; `ComparisonControl/TargetGroups` is the `rdfs:label`. |
| **A no-intervention arm** | `Treated_Untreated` < `ComparisonControl_TargetGroups` | The nearest EXPO has to a control arm. |
| **Graded intervention strength** | `DoseResponse` | "A target group receives a specified dose of a specified treatment." |
| **A staged schedule** | `TimeCourse` < `ComparisonControl_TargetGroups` | |
| **Groups split from a common origin** | `SplitSamples` < `PairedComparisonOfMatchingGroups` | Closest term to branches forked from a shared ancestor state. |
| Staged design | `SequentialDesign` < `ExperimentalDesign` | |
| What is being compared | `MethodsComparison`, `SubjectsComparison` | Plural. The singular forms never shipped. |
| Normalisation / quality control | `NormalizationStrategy`, `QualityControl` | Not `QualityControlStrategy`. |
| A comparison that was invalid | `FaultyComparison`, `ErrorOfConclusion`, `IncompleteDataError` | First-class. Use them. |

**Confirmed absent from all 324 classes: `Control`, `Replication`, `ReferenceExperiment`.** Those names circulate in mapping tables and are refuted. Note the trap in the other direction: `IndependentVariable` and `DependentVariable` **do** exist — reporting them absent (as circulated corrections sometimes do) costs you real vocabulary.

## The disjointness constraint — read before you model

EXPO declares its design strategies mutually `owl:disjointWith`. Verified in the OWL:

- `TimeCourse` ⊥ `DoseResponse`, `Treated_Untreated`, `Normal_Disease`, `GeneKnock-in`, `GeneKnock-out`
- `Treated_Untreated` ⊥ `DoseResponse`, `Normal_Disease` and the `GeneKnock-*` classes
- `ComparisonControl_TargetGroups` ⊥ `PairedComparisonOfMatchingGroups`, `PairedComparisonOfSingleSampleGroups`
- `PairedComparison` ⊥ `QualityControl`; `NormalizationStrategy` ⊥ both
- `SplitSamples` has **no declared** disjointness with the three above — its only declared pair is `SymmetricalMatching`. But `SplitSamples < PairedComparisonOfMatchingGroups`, and that class *is* declared disjoint with `ComparisonControl_TargetGroups`, so the conclusion follows by inheritance rather than by a direct axiom. Say which it is; a reader who greps for a `SplitSamples`/`DoseResponse` axiom and finds none will assume you fabricated it.

**A paired-branch intervention pipeline needs several of these at once** — a no-intervention arm (`Treated_Untreated`), a strength ramp (`DoseResponse`), a staged schedule (`TimeCourse`), branches from a common origin (`SplitSamples`). EXPO's own axioms forbid one strategy instance being more than one of them. Assert it and **a reasoner reports your ontology inconsistent.**

The fix is not to pick one. Attach **several distinct `ExperimentalDesignStrategy` instances** to a single `ExperimentalDesign`. That is legal, and it forces you to state each strategy separately — precisely what a one-row-per-concept mapping table hides.

**A second consequence worth stating out loud:** if you vary intervention strength *and* withdraw a scaffold on the same schedule, the effects are confounded, and your design is multi-factor rather than a single comparison. EXPO has names for the resulting defects — `FaultyComparison`, and `ErrorOfConclusion` if a conclusion is drawn from it (siblings under `ResultError`, not a chain).

## The two genuine gaps

### Gap 1 — a control that is not an `Object`

SUMO's MILO defines `experimentalControl` as a `CaseRole`, a subrelation of `patient`, with `(domain experimentalControl 2 Object)`. Argument 2 must be **physical**.

For a wet-lab control — a control sample, an untreated plate — that is right. For a computational experiment whose control is a **no-intervention branch**, it is not: a branch is a process (or the plan describing one), not an object. The predicate does not fit its own domain.

**Mint.** Define your own control relation whose range is the entity you actually have, and document the relationship to `experimentalControl` in prose rather than asserting a subproperty axiom you cannot justify. Record it in the gap register.

### Gap 2 — replication and the unit of analysis

EXPO has no `Replication` term at all. This is the more consequential gap, because replication is not one concept — it is a family, and conflating its members is how formalisations end up licensing statistical claims they cannot support.

**Mint three distinguishable things, not one:**

| Kind | What is held fixed | What varies | What it licenses |
|---|---|---|---|
| **Exact repetition** | Everything recorded, including seeds | Nothing intentionally | Determinism claims. If results differ, your record is incomplete. |
| **Controlled replication** | The common starting state and the environment | One factor, deliberately | Causal attribution to that factor. |
| **Independent replication** | The protocol only | Starting state, environment, operator | Generalisation beyond one setting. |

**And mint the unit of analysis explicitly.** This is the single most important field in the whole comparison layer:

> **The unit of analysis is the thing you have N of.**

Branches forked from a **common ancestor state** and fed **identical subsequent inputs** are *paired observations of one unit*, not independent samples. A formalisation that records them as N independent runs poisons the downstream analysis in one of two directions, and it is worth knowing which:

- **Pairing dropped, pooling kept** — the analysis compares arms without knowing they share an ancestor. Ancestor variance that would have cancelled in the paired contrast is left in the error term, intervals come out *too wide*, and real effects are missed. The design's whole advantage is thrown away.
- **Branch count taken as the unit count** — the analysis treats *m* branches from one ancestor as *m* independent units. This is pseudoreplication: intervals come out *too narrow*, and under any selection pressure that error is banked in the intervention's favour.

Either way the record has licensed a claim it cannot support. Only a recorded pairing identifier lets the analysis get both the contrast and the unit count right.

Make the record say so:

- an identifier for the **common ancestor state** every arm forked from,
- an identifier for the **shared input sequence** the arms received,
- an explicit **arm role** (intervention / no-intervention / reference),
- and a declared **unit of analysis** naming which identifier defines independence.

If two arms share an ancestor identifier, they are paired. That is now a machine-checkable fact rather than something a reader must infer.

**For the statistics that follow from this, load `/counterfactual-statistics`.** This sheet gets the structure into the record; that pack tells you what may legitimately be concluded from it.

## The no-intervention arm is a formalisation invariant

**A comparison structure with no "do nothing" arm is incomplete, and the critic flags it as such.**

Not because doing nothing is interesting, but because without it the experiment cannot distinguish *the intervention worked* from *things improved anyway*. In any system that intervenes repeatedly on a moving baseline, the no-intervention arm is the cheapest and least arguable defence against a permanent record of confirmation.

Model it as a `FactorLevel` — the null level of the factor — not as a separate special-cased entity. It then falls out of the design vocabulary rather than being bolted on, and it cannot be quietly omitted without leaving a hole in the factor's level set.

Three properties the record must make checkable:

1. **Presence** — the null level exists for every comparison.
2. **Pairing** — it forked from the same ancestor and received the same inputs as the intervention arms.
3. **Eligibility to win** — the record can represent "the no-intervention arm was better." If your schema has no way to express that outcome, you have built an apparatus that cannot report failure.

Property 3 is the one that gets designed away, usually by making the decision record's outcome enum `{adopt, defer}`.

## Reference arms

Arms running a well-understood conventional alternative — the standard method, last quarter's model, a known-good baseline. EXPO has no `ReferenceExperiment`. **Split the two senses that name conflates:** reference *arms* are objects under `GroupExperimentalObject` / `ObjectOfExperiment`; published baselines you compare against are `BiblioReference` / `DBReference`. `MethodsComparison` covers the comparison itself.

Two distinctions worth recording:

- **Reference vs. control.** A control receives *no* intervention. A reference receives a *different, known* one. Collapsing them destroys the meaning of both.
- **Reference vs. scaffold.** Reference arms are permanent measuring instruments. Scaffolds are temporary supports that get withdrawn (see [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md)). The same artifact can serve both roles at different times — which is exactly why the *role* belongs on the relation, not baked into the artifact's class.

## Blinding, if the comparison is adjudicated

If some component judges arms and must not know which arm came from where, the record layer must support that. **Structural blinding, not promised blinding:** the blinded view is a projection that does not contain the field, rather than a field the reader agrees not to look at.

The ontology layer can *describe* which view was used and assert that a decision was made from a blinded projection. It cannot *enforce* it — enforcement lives in the contracts. That division is the projection law. For the enforcement side, load `/contract-engineering` (`blinding-by-construction`).

## Worked shape

A generic paired-branch intervention pipeline. Names are illustrative; the structure is the point.

```
Comparison
├── ancestor_state_id        ← identity of the common fork point   (mint)
├── input_sequence_id        ← identity of the shared future inputs (mint)
├── unit_of_analysis         ← "ancestor_state_id"                  (mint)
├── factor                   ← EXPO: ExperimentalFactor
│   └── levels               ← EXPO: FactorLevel
│       ├── none             ← the no-intervention level (REQUIRED)
│       ├── candidate_A
│       └── reference_conventional
├── target_variables         ← EXPO: TargetVariable  (utility, stability, cost)
├── design_strategies[]      ← EXPO: SEVERAL instances — Treated_Untreated,
│                              DoseResponse, TimeCourse, SplitSamples — never one:
│                              three are pairwise disjoint by declared axiom, the
│                              SplitSamples pairs by inheritance
├── arms[]
│   ├── arm_role             ← intervention | none | reference      (mint)
│   ├── realised_by          ← SUMO: realization → the Process that ran
│   └── results              ← EXPO: ExperimentalResults
└── result_errors[]          ← EXPO: ResultError / FaultyComparison / IncompleteDataError
```

Note what is inherited and what is minted. Roughly half is EXPO vocabulary used correctly; the mints are few, named, and each traceable to a specific gap. **That ratio is what a healthy extension looks like** — see [`extending-without-forking.md`](extending-without-forking.md). If you are minting most of your terms, you have not read the source carefully enough. If you are minting none, you are probably citing terms that do not exist.

## Competency questions this layer must answer

Write these as executable queries before you model anything ([`competency-questions-first.md`](competency-questions-first.md)):

1. Was this intervention compared against a no-intervention arm? *(presence)*
2. Did the compared arms fork from the same ancestor state and receive the same inputs? *(pairing)*
3. How many independent units underlie this claim — and by which identifier? *(unit of analysis)*
4. Which arms were reference arms, and which conventional alternative did each run?
5. Has any comparison been marked as faulty, and on what grounds?
6. Were there comparisons where the no-intervention arm won — and are they retained? *(if this returns zero rows, suspect the record, not the world)*

**Question 6 is the health check for the whole layer.** A system that has never recorded a null result is either extraordinarily lucky or is not recording them.

## Checklist

- [ ] `ExperimentalFactor` / `FactorLevel` / `TargetVariable` used — names taken from the OWL, not the paper's figure.
- [ ] Design strategies attached as several disjoint instances, not collapsed into one.
- [ ] The no-intervention arm exists as the null `FactorLevel` of the factor.
- [ ] Ancestor-state and input-sequence identifiers recorded on every arm.
- [ ] Unit of analysis declared explicitly, naming the identifier that defines independence.
- [ ] Arm role (intervention / none / reference) is a recorded field, not inferred from a name.
- [ ] The outcome vocabulary can express "the no-intervention arm won."
- [ ] Replication kind distinguished (exact / controlled / independent) where replication is claimed.
- [ ] Control-relation mint recorded in the gap register with its reason (the `Object` domain constraint).
- [ ] `FaultyComparison` / `IncompleteDataError` reachable from the result record.
