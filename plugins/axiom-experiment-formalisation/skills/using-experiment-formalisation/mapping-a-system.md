# Mapping a System

**The mapping table is an intermediate artifact, never a deliverable.** Handing someone a two-column table of "your concept → ontology concept" and calling the formalisation done is the most common failure in this entire discipline. The table cannot be executed, cannot be tested, cannot be wrong in any detectable way, and — as circulated tables reliably demonstrate — is frequently full of class names that do not exist.

This sheet gives the procedure that produces a table worth having, and says what must come after it.

## Before you map

Two prerequisites. Skipping either produces a table that looks fine and serves nothing.

1. **Competency questions exist**, prioritised, with drafted queries for the Musts ([`competency-questions-first.md`](competency-questions-first.md)). Without them you cannot tell whether a mapping is adequate — there is no criterion.
2. **The tier is chosen** ([`formalisation-triage.md`](formalisation-triage.md)). Tier 1 maps to a JSON-LD context over records you already emit; Tier 2 maps to an OWL module. The mapping is the same shape either way, but the destination differs and it is cheaper to know first.

## Step 1 — Inventory what you actually have

Not what the architecture document says. What is emitted.

Enumerate:

- **Record types** that cross a boundary — the typed schemas, their fields, their versions.
- **Activities** — the stages that consume and produce those records.
- **Agents** — the components, services, models, or people that perform activities.
- **Roles** — which agent does what in which activity, and which separations are load-bearing.
- **Identities** — what is content-addressed, what has a surrogate key, what has neither.
- **Comparison structure** — arms, ancestors, shared inputs, units of analysis.
- **Absence encoding** — how "not measured" is currently represented, honestly.

**Read the code, not the design document.** The gap between them is usually where the interesting findings are, and a mapping built on the document maps a system that does not exist.

## Step 2 — Candidate terms

For each inventory item, walk the mint ladder ([`extending-without-forking.md`](extending-without-forking.md)): reuse → narrow → annotate → mint. Search in order: EXPO, PROV-O, your upper ontology, domain ontologies, [`prior-art-map.md`](prior-art-map.md).

Propose a candidate term, or explicitly propose a mint. Do not leave a row blank — a blank row is an unmade decision that will be made badly later.

## Step 3 — Verify every candidate

**The step that distinguishes this procedure from the tables people circulate.**

For each candidate external term, confirm it exists in the source you intend to cite ([`expo-verified-inventory.md`](expo-verified-inventory.md), sixty-second procedure). Record the outcome in the row itself:

| Verification | Meaning |
|---|---|
| **Verified** | Found in a primary source. Row records the source and location. |
| **Unverified** | Could not be confirmed. **Not** the same as false — say so precisely. Either verify it or treat it as a mint. |
| **Refuted** | Checked and absent from the source it was attributed to. Record it in the do-not-cite list so nobody re-proposes it. |

**A row may not proceed to the artifact stage while any external term in it is Unverified.** That is the gate.

## Step 4 — Fit verdicts

Verification asks *does this term exist?* Fit asks *does it mean what I mean?* Both are required; the second is where the modelling actually happens.

| Verdict | Meaning | Action |
|---|---|---|
| **Exact** | The term means what you mean. | Reuse. |
| **Narrower** | Yours is a kind of theirs; every instance of yours is one of theirs. | Subclass. Test that sentence literally. |
| **Broader** | Theirs is a kind of yours. | Do **not** subclass. Either mint your general term, or accept theirs and lose generality — deliberately. |
| **Overlapping** | Neither contains the other. | Mint. Document the overlap in prose. Overlap asserted as subclass is how ontologies become inconsistent. |
| **Constrained** | Right meaning, wrong constraints — domain, range, or cardinality forbid your use. | Mint a sibling and document the divergence (the `experimentalControl` case, [`sumo-upper-binding.md`](sumo-upper-binding.md)) — **or**, where the constraint is structural rather than semantic, reuse the term via the annotate rung of the mint ladder. |
| **No fit** | Nothing close. | Mint. Record in the gap register. |

**"Overlapping" and "Constrained" are the verdicts that get skipped**, and both get recorded as "Exact" by someone in a hurry. That is how a mapping becomes wrong while every row looks green.

**Be willing to record a bad fit.** A row reading *"our judge stage — EXPO has no adjudication concept; `DataAnalysis` was proposed and is refuted; minting"* is a better artifact than a green row that quietly means nothing.

## Step 5 — The gap register

Every mint gets a row ([`extending-without-forking.md`](extending-without-forking.md)): the gap, why nothing fits, the competency question, the resolution, the module, the status. The register is the mapping's twin and reviewers should read them together.

## Step 6 — Produce the artifacts

**This is where a mapping becomes a formalisation.** Minimum for Tier 1:

1. **A JSON-LD context** binding your existing record fields to the mapped terms — so the records you already emit become interpretable without changing the producers.
2. **The competency-question queries**, executable, with their controls.
3. **The gap register.**
4. **The sync check** wiring ontology terms to contract fields ([`the-projection-law.md`](the-projection-law.md)).
5. **The verified-terms record** — which terms, verified against what, on what date.

Tier 2 adds the OWL module, SHACL shapes, and the reasoner checks ([`validation-and-conformance.md`](validation-and-conformance.md)).

**If the output stops at the table, the work has not been done.** The table is scaffolding for the artifacts; it is not the thing.

## Row shape

**The CQ-x labels below are local to this worked example** — they do not correspond to the numbered set in [`competency-questions-first.md`](competency-questions-first.md). Number yours once, in one place, and make every sheet cite that set.

Nine columns. It is worth all nine — a narrower table hides exactly the information reviewers need.

| Our concept | Where it lives (real path) | Candidate term | Source | Verification | Fit verdict | Resolution | CQ | Notes |
|---|---|---|---|---|---|---|---|---|
| the campaign | `runner/campaign.py` | `ComputerSimulation` < `ComputationalExperiment` | expo.owl | Verified | Exact | reuse | CQ-1 | figure prints `ComputationalExp.`; cite the fragment |
| branch execution | `runner/branch.py:Branch` | `ExecutionOfExperiment` | expo.owl | Verified | Exact | reuse | CQ-1 | not `ComputationalExperiment` — that is the design, this is the run |
| no-intervention arm | `runner/branch.py:Arm.role` | `Treated_Untreated` < `ComparisonControl_TargetGroups` | expo.owl | Verified | Constrained | reuse via annotate rung (arm role) | CQ-1 | proposed `Control` is **refuted**; MILO `experimentalControl` domain 2 = `Object`, does not fit a process |
| candidate variable | `spec/factor.py` | `ExperimentalFactor` / `FactorLevel` | expo.owl | Verified | Exact | reuse | CQ-1 | **not `Factor`** — that name is in the paper and not in the OWL |
| "this factor is independent" | `spec/factor.py:independent` | `IndependentVariable` < `Independence` | expo.owl | Verified | **Constrained** | reuse as attribute | CQ-1 | it is an attribute *of* a variable, not a variable type |
| measured outcome | `results/metrics.py` | `TargetVariable` | expo.owl | Verified | Exact | reuse | CQ-2 | |
| evidence record | `qa/report.py:Report` | `ExperimentalResults` < `ProductRole` | expo.owl | Verified | Narrower | subclass | CQ-3 | proposed `ResultSet` is refuted; note it is a **role**, not an object |
| decision | `judge/decision.py` | `ResultsEvaluation` / `ResultsInterpretation` | expo.owl | Verified | Overlapping | reuse + mint | CQ-3 | proposed `DataAnalysis` is refuted; these are the real evaluation classes |
| lineage | `store/graph.py` | `prov:wasDerivedFrom` | PROV-O | Verified | Exact | reuse | CQ-4 | proposed `Lineage` is refuted |
| who did what | `store/graph.py` | `prov:hadRole` | PROV-O | Verified | Exact | reuse | CQ-6 | roles as relations, not classes |
| authorisation | `judge/token.py` | — | — | — | No fit | mint | CQ-5 | absent from EXPO and SUMO |

Read that table's shape: **roughly half reuse, a handful of refutations, a few disciplined mints, every row traceable to a question and a real path.** That ratio is the health signal. Mostly mints means the sources were not searched. No refutations against a proposed table means verification was nominal. No file paths means the mapping was built from a document rather than a system.

## Review gate

- [ ] Every row cites a real path in the system, not a document.
- [ ] No external term is Unverified.
- [ ] Every refuted term is recorded in the do-not-cite list.
- [ ] Every row has a fit verdict, and Overlapping/Constrained were genuinely considered.
- [ ] Every mint has a gap-register row and a competency question.
- [ ] The artifacts of step 6 exist — the table is not the output.
- [ ] At least one competency question is now answerable that was not before.
