# EXPO: The Verified Inventory

**Law 1 of this pack: a term is cited as EXPO or SUMO vocabulary only if you have checked it in a primary source. Everything else is declared as your extension.**

And the corollary this sheet exists to teach, learned the hard way:

> **The paper is not the ontology. The figure is not the ontology. The ontology is the ontology.**

## What EXPO is

The generic ontology of scientific experiments: Soldatova & King (2006), *An ontology of scientific experiments*, J. R. Soc. Interface 3(11):795–803, [doi:10.1098/rsif.2006.0134](https://doi.org/10.1098/rsif.2006.0134). Open access.

It formalises what is common to *all* experiments — goals, design, model, planned actions, hypotheses, results, typed errors, administrative metadata — deliberately above any domain. Its definition of the thing it models:

> "a research method which permits the investigation of cause-effect relations between known and unknown (target) variables of the domain."

**The shipped artifact** (`expo.owl`, sha256 `57ede339…c99847`, namespace `http://www.hozo.jp/owl/EXPOApr19.xml/`) contains **324 classes and 78 object properties**, no datatype properties, and — matching the paper's stated rule — no multiple inheritance. Authored in Hozo, exported to OWL.

**Why it is still worth using twenty years on:** nothing has replaced it at that level of generality. OBI is larger and maintained but biomedical; ISA is a file format; the ML-experiment ontologies are dormant ([`prior-art-map.md`](prior-art-map.md)). EXPO remains the best published articulation of the *grammar* of an experiment.

**Adopt it for the right reason.** EXPO is routinely proposed as a route to *interoperability* — and it does not deliver that. Essentially nobody consumes EXPO; there is no ecosystem of tools reading it, and "we applied EXPO, therefore we are interoperable" is a claim that fails in month four. Interoperability comes from RO-Crate for packaging, Croissant for ML datasets, PROV-O for lineage, and public pre-registration of design and thresholds — in roughly that order of payoff.

What EXPO uniquely gives you is **methodological precision**: a published, citable vocabulary for paired comparison against a control, under a factor model, with a named taxonomy of the ways a result can be wrong. Cite it so an outside methodologist can audit your design without reading your code. That is worth a great deal, and it is a different thing from machine exchange.

## Getting the artifact

`expo.sourceforge.net` shows the project; the file archive at `sourceforge.net/projects/expo/files/` returns **403 to automated clients**, which is why EXPO is widely and wrongly reported as unobtainable. **It is obtainable.** Retrieve it through a browser, or from a research mirror, and verify the hash.

Two obligations before you cite it in anything that matters:

1. **Byte-verify against the official archive**, downloaded through a browser, if class names are load-bearing for a publication.
2. **Cite the paper for design rationale and the OWL for names.** They disagree, and the OWL wins.

**Two reported discrepancies, both reconciled** — recorded here so they are not re-litigated:

- *"The archive is ~36 KB, but your file is 349 KB."* 36 KB is the **compressed** size. This file deflates to 36,191 bytes. Same artifact.
- *"Third-party tooling reports 347 classes, the paper says 218, you say 324."* All three are counting different things. **324** is the number of `owl:Class` declarations in the EXPO namespace — the defensible figure. The file contains **402** distinct EXPO-namespace IRIs in total: those 324 classes plus 78 object properties, with **zero referenced-but-undeclared** IRIs. The paper's **218** describes the initial version, which the shipped April 2006 file post-dates. **We could not reproduce 347 from the shipped file** under any counting we tried — if you meet that number, ask what was counted rather than assuming either source is wrong.

**Note the shape of that second item.** Sources give different numbers because they count different populations. When a count matters, state what you counted — and say plainly when you cannot reconstruct someone else's figure, rather than inventing a mechanism that would explain it. An earlier draft of this sheet did exactly that, asserting a "326 referenced IRIs, two undeclared" reconciliation that does not survive a scan of the file.

**SourceForge rate-limits with HTTP 200 and a 2-byte body.** If you script the retrieval, assert on file size and zip magic bytes — not on the exit code.

## The paper–OWL discrepancy

This is not a footnote. **Roughly ten names printed in the paper never shipped**, and several shipped names contain typos nobody would guess. An inventory built from the published figure — which is what the first version of this very sheet was — is wrong in about a dozen places.

| Written by almost everyone | Actual OWL fragment |
|---|---|
| `Factor` | **`ExperimentalFactor`** (< `Variable`) |
| `AdminInfoAboutExperiment` | **`AdminInfoExperiment`** |
| `PlanExperimentalActions` | **`PlanOfExperimentalActions`** |
| `QualityControlStrategy` | **`QualityControl`** |
| `TitleOfExperiment` | **`TitleExperiment`** |
| `MethodComparison` / `SubjectComparison` | **`MethodsComparison`** / **`SubjectsComparison`** |
| `ParedComparison` (figure's typo) | **`PairedComparison`** |
| `FalsePositive` | **`FalsePpositive`** *(sic — misspelled in the OWL; `FalseNegative` is correct)* |
| `ComparisonControl/TargetGroups` | fragment **`ComparisonControl_TargetGroups`** (that slash string is the `rdfs:label`) |
| `GalileanExp` / `BaconianExp` | **`GalileanExperiment`** / **`BaconianExperiment`** |
| `ComputationalExp.` | **`ComputationalExperiment`** (and **`ComputerSimulation`** beneath it) |

Other misspellings live in the shipped file: `PoblemAnalysis`, `InformationGethering`, `RecommendationSatus`, `EperimentalDesignTask`, and a property named `has_pasword`. **Reproduce them exactly.** A "corrected" citation is a citation to nothing.

Four classes carry an `rdfs:label` differing from the fragment (`Treated_Untreated`, `ComparisonControl_TargetGroups`, `Normal_Disease`, `DDC_Dewey_Classification`). **Use the fragment as the name**; quote the label only as a label.

## Present, though widely reported absent

The mirror image of hallucination, and just as damaging. All four are **in the shipped OWL**:

| Term | Reality |
|---|---|
| `IndependentVariable` | **Present — but not a kind of variable.** `IndependentVariable < Independence < AttributeOfVariable`. It is an *attribute you attach to* a variable. The variable is `ExperimentalFactor`. (`Independence` is disjoint with `Controllability`.) |
| `DependentVariable` | **Present — and, like `IndependentVariable`, it is an attribute**: `DependentVariable < Independence < AttributeOfVariable`. The measured-outcome *class* is `TargetVariable < Variable`. |
| `ExperimentalProtocol` | **Present.** |
| `ExperimentalObservation` | **Present** — and it is a *role*: `< ProductRole < Process-relatedRole < Role`. |

**This is why *unverified* and *refuted* must never be collapsed** ([`the-projection-law.md`](the-projection-law.md)). Reporting "not in EXPO" for something you merely could not find in a figure is the same error as citing something you never checked — it just fails in the other direction, and it costs you real vocabulary.

## Confirmed absent

Checked against all 324 classes. These are **refuted**, not merely unverified:

`Control` · `Replication` · `ReferenceExperiment` · `ResultSet` · `ExperimentalResult` (singular) · `Measurement` · `ExperimentRecord` · `Lineage` · `DataAnalysis` · `HypothesisTesting`

Full list with the correct move for each: [`verified-terms.json`](verified-terms.json).

## Structural facts that change how you model

Naming is the shallow half. These are the findings that change the design.

### Results are roles, not objects

`ExperimentalResults`, `ExperimentalObservation`, and `Error` all sit under `ProductRole < Process-relatedRole < Role` — under `Abstract`, not `Physical`. **In EXPO an evidence record is a role that data plays in a process, not a stored artifact.**

If your schema treats a result as a thing on disk, that is a genuine impedance mismatch. Resolve it deliberately: either adopt the role framing, or record that you diverge and why. Do not discover it after the mapping ships.

Meanwhile `ExecutionOfExperiment < ScientificActivity` and `ExperimentalAction < ExecutionOfExperiment` are the process side — so a branch and a step within it share one subtree, with no inheritance conflict.

### Design strategies are disjoint

**The finding most likely to break a real ontology.** Verified `owl:disjointWith` axioms include:

- `TimeCourse` ⊥ `DoseResponse`, `Treated_Untreated`, `Normal_Disease`, `GeneKnock-in`, `GeneKnock-out`
- `Treated_Untreated` ⊥ `DoseResponse` and the `GeneKnock-*` classes
- `ComparisonControl_TargetGroups` ⊥ `PairedComparisonOfMatchingGroups`, `PairedComparisonOfSingleSampleGroups`
- `PairedComparison` ⊥ `QualityControl`; `NormalizationStrategy` ⊥ both

**Consequence:** a single `ExperimentalDesignStrategy` instance cannot be several of these at once. A design that needs a no-intervention control *and* a graded intervention strength *and* a staged schedule *and* branches split from a common origin cannot be one strategy. Assert it as one and **a reasoner will report your ontology inconsistent.**

The fix is not to pick one. Attach **several distinct strategy instances** to a single `ExperimentalDesign`. That is legal, and it forces each strategy to be stated separately — which is exactly what a one-row-per-concept mapping table hides. See [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md).

### The error subtree is richer than the figure shows

`Error < ProductRole` roots the whole subtree, and it is deeper than it looks:

```
Error
├── ObservationalError
│   └── MeasurementError
│       ├── RandomError → StatisticalError
│       ├── SystematicError
│       └── CoarseError
└── ResultError
    ├── ErrorOfConclusion
    ├── FaultyComparison
    ├── IncompleteDataError
    └── HypothesisAcceptanceMistake → FalsePpositive (sic) / FalseNegative
```

Two things people get wrong. `ResultError` is **under** `Error`, not separate from it. And `ErrorOfConclusion` and `FaultyComparison` are **siblings, not a chain** — write `FaultyComparison` < `ResultError`, never `FaultyComparison` → `ErrorOfConclusion`.

**EXPO types the ways a result can be wrong.** Widely overlooked, and unusually valuable: if your formalisation cannot express "this conclusion rests on a faulty comparison" or "this rests on incomplete data," you have discarded something EXPO gave you for free.

## Terms worth knowing for computational work

Easy to miss, and unusually good fits:

| Term | Use |
|---|---|
| `Treated_Untreated` < `ComparisonControl_TargetGroups` | The nearest EXPO has to a no-intervention control arm. |
| `DoseResponse` | A target group receiving a specified dose — the natural home for graded or partial intervention strength. |
| `TimeCourse` | Staged schedules over time. |
| `SplitSamples` | Groups split from a common origin — closest to branches forked from a shared ancestor state. |
| `SequentialDesign` < `ExperimentalDesign` | Staged designs. |
| `ComputerSimulation` < `ComputationalExperiment` | Often a closer fit than `ComputationalExperiment` alone. |
| `ResultsEvaluation`, `ResultsInterpretation` | The evaluation and interpretation tasks — use these instead of the non-existent `DataAnalysis`. |
| `FactSupportResearchHypothesis`, `FactRejectResearchHypothesis` | Outcome classes. |
| `LevelOfSignificance` < `StatisticsCharacteristic` | |
| `PermissionStatus` < `StatusExperimentalDocument` | **Document access permission only** — not a delegated authorisation from an adjudicator. Thin. Pair with PROV-O for the issuing act. |

## EXPO's design rules — copy three of them

| Rule | What EXPO did | Copy? |
|---|---|---|
| **Minimal relations** | 78 object properties for 324 classes, and the paper describes an even smaller conceptual set (`subclass`, `instance-of`, `part-of`, `attribute-of`, plus `Role`). | **Yes.** Relation proliferation is how extensions become unmaintainable. |
| **No multiple inheritance** | Deliberate, following the Foundational Model of Anatomy. **Zero** classes have more than one parent: 323 have exactly one, and the root has none. | **Yes** for a first extension. |
| **No individuals** | EXPO contains no instances; particular experiments come from extensions. | **Yes** — and this is the one broken first. If your OWL file grows when you run an experiment, you have made this mistake. |
| **Three levels** | Physical (`FieldOfStudy`) / model (`ExperimentalModel`) / design (`ExperimentalDesign`). | **Yes.** The cleanest available answer to "is this the world, our theory of it, or our plan?" |

## Galilean and Baconian

`GalileanExperiment` (hypothesis-driven) versus `BaconianExperiment` (hypothesis-forming). EXPO also ships `ExplicitHypothesis` / `ImplicitHypothesis`.

**Two verified quirks of the shipped OWL before you use these.** This is the sheet's own banner applied to itself — the paper describes a tidy pairing, and the artifact does something else:

1. **Both are under `PhysicalExperiment`.** `GalileanExperiment < PhysicalExperiment` and `BaconianExperiment < PhysicalExperiment`, and `PhysicalExperiment` is a *sibling* of `ComputationalExperiment`. So classifying a computational pipeline as a `BaconianExperiment` entails it is a physical experiment. Do not assert it; declare the mode as your own attribute and cite EXPO's distinction as the source of the concept.
2. **`Hypothesis-formingExperiment` is a child of `GalileanExperiment`**, not of `BaconianExperiment` — as is `Hypothesis-drivenExperiment`. `BaconianExperiment` has no children at all. The OWL's structure contradicts the paper's Galilean-equals-driven / Baconian-equals-forming pairing.

The *concept* is sound and worth adopting. The *class assertions* are not usable as-is for computational work.

This matters operationally because search- and generation-based pipelines are largely Baconian, and reviewers routinely read that as sloppiness. It is not — it is a **declared mode**, and EXPO gives you vocabulary to declare it. What is not permissible is *drifting*: running exploratory and reporting hypothesis-driven, i.e. writing the hypothesis after seeing the results.

**Make the mode a required field, recorded before execution.** Pair with `/counterfactual-statistics` for the consequences (multiplicity, the winner's curse, pre-registration).

## Verifying a term in sixty seconds

1. **Is it in [`expo-owl-inventory.json`](expo-owl-inventory.json)?** That file is extracted mechanically from the OWL and is authoritative for names. If yes, cite the fragment exactly.
2. **Is it in the corrections or do-not-cite list in [`verified-terms.json`](verified-terms.json)?** Apply the recorded move.

   Check `subclass_of` too, not just the name — the position is as citable as the spelling, and it is where the surprises live (`IndependentVariable` is an attribute; `ResultError` is under `Error`; `SplitSamples` is under `PairedComparisonOfMatchingGroups`). The `disjoint_with` map is **one-directional**: to test whether A and B are disjoint, look up both keys.
3. **Neither?** Grep the OWL itself. Found → add it to the inventory. Not found → it is **refuted for EXPO**; mint in your namespace and record the gap.

For SUMO, the same procedure against `Merge.kif` and `Mid-level-ontology.kif` — and remember those are not all of SUMO, so "not found" there is *unverified*, not refuted.

## What EXPO genuinely lacks

Real gaps, each a legitimate mint:

| Gap | Sheet |
|---|---|
| Provenance and lineage | [`provenance-and-lineage.md`](provenance-and-lineage.md) |
| Replication kinds and the unit of analysis | [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md) |
| Epistemic role separation and its prohibitions | [`governance-role-extensions.md`](governance-role-extensions.md) |
| Delegated authorisation (`PermissionStatus` is document access, not a warrant) | [`governance-role-extensions.md`](governance-role-extensions.md) |
| Lifecycle states and transitions | [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md) |
| Reversibility of a graded intervention (`DoseResponse` covers strength, not reversal) | [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md) |
| Scaffold withdrawal gates (`SequentialDesign`/`TimeCourse` stage, but do not gate) | [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md) |
| Explicit absence semantics | [`measurement-uncertainty-and-absence.md`](measurement-uncertainty-and-absence.md) |

**Record every gap in the gap register.** It turns "EXPO didn't cover this" from an excuse into a design artifact — see [`extending-without-forking.md`](extending-without-forking.md).

## Maintenance reality

EXPO is **unmaintained**. The shipped OWL contains at least five misspelled class names and a property named `has_`. It is entirely citable as a conceptual framework and a source of vocabulary and structure. It is not a maintained standard you can take a dependency on, and no one is going to fix the typos.

Model accordingly: re-express in your own namespace, cite the paper for rationale and the OWL fragments for names, and never assert `owl:equivalentClass` against an IRI whose host you do not control.
