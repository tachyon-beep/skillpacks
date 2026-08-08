# The Projection Law

**Read this before anything else in the pack.** Everything else is technique; these are the two rules that decide whether the technique produces an asset or a liability.

---

## Law 1 — Verified vocabulary only

> **A term is presented as EXPO, SUMO, PROV-O, or any other external vocabulary only if it has been checked in a primary source. Every other term is declared as your own extension.**

Not "sounds right." Not "a model said so." Not "the consultant's table had it." Checked, in the source, with the location recorded.

**Why this is Law 1 rather than a tip:** the failure is silent and durable. A hallucinated class name is well-formed, plausible, and consistent with the surrounding real terms. It passes review because nobody checks. It reaches a schema, a paper, a JSON-LD context. Then someone downstream tries to resolve it, and the entire mapping's credibility collapses at once — including the parts that were correct.

The mapping tables that circulate for this task are demonstrably full of them: `Control`, `Replication`, `ReferenceExperiment`, `ResultSet`, `ExperimentRecord`, `Lineage`, `DataAnalysis`, `HypothesisTesting` are attributed to EXPO and appear nowhere in its 324 classes. `Agent` is attributed to SUMO, whose class is `AutonomousAgent`. `Hypothesis` is attributed to SUMO, which has no such term in its core or mid-level files.

**And the error runs the other way too.** An earlier version of this pack, built from the published paper's figure rather than the shipped OWL, reported `IndependentVariable`, `ExperimentalProtocol`, and `ExperimentalObservation` as absent. All three exist. It also cited `Factor`, `AdminInfoAboutExperiment`, and `QualityControlStrategy` — names that appear in the paper and never shipped. **The paper is a secondary source for names, even though the authors wrote both.** Only parsing the artifact settles it.

### The three verdicts

Precision here matters, because overclaiming in the other direction is its own error:

| Verdict | Means | Say |
|---|---|---|
| **Verified** | Found in a primary source, location recorded. | "EXPO `TargetVariable` (expo.owl; `expo-owl-inventory.json`)." |
| **Unverified** | Not found, but the source is partial or was not exhaustively searched. | "I could not verify this against the paper." |
| **Refuted** | Checked where it must appear if real; absent. | "Not an EXPO class." |

**Never collapse Unverified into Refuted.** EXPO's published figure is a hand-drawn fragment of an ontology the paper sizes at 218 concepts, and the shipped OWL declares 324 — absence from the figure is weak evidence about either. Conversely, never let Unverified drift into a citation.

### Operationally

- [`verified-terms.json`](verified-terms.json) is the inventory; the `formalisation-critic` loads it.
- Adding a term to the inventory requires a source and a date.
- A term that is Unverified is either verified before use, or minted in your own namespace and recorded as a gap.
- A refuted term goes on the do-not-cite list with its correct replacement, so it does not get re-proposed next quarter.

---

## Law 2 — The projection law

> **The ontology describes your typed contracts. It never becomes the runtime source of truth. When the two disagree, the contract wins and the ontology is corrected.**

The formalisation is a **projection**: a queryable, interoperable, auditable *description* of records whose authority lives elsewhere. Fail-closed enforcement, authority boundaries, and identity live in the contract layer. The graph tells you what happened. It never decides what may happen.

### Why this is a law and not a preference

Every argument for inverting it sounds good in the room:

- *"Declarative policy in SHACL is more auditable than code."* — It is more *readable*. It is also now a second implementation of your decision logic, in a language with different absence semantics, evaluated by a component with different availability characteristics.
- *"Retire the typed schemas so we don't maintain two models."* — The typed schemas are what makes absence loud, versions fail closed, and authority enforceable. OWL has open-world semantics: a missing required property is not an error, it is an incomplete description. SHACL can restore closed-world cardinality, but only over the nodes its targets select — so retiring the contracts trades a parser that fails closed on every record for a validator that fails open on any record it does not select.
- *"The knowledge graph should be the single source of truth."* — Single source of truth is right. The graph is the wrong candidate: it is derived, eventually consistent, and reconstructible. Make the contracts authoritative and the graph reproducible from them.
- *"But SHACL is closed-world, so it can enforce required fields."* — True, and it is the strongest form of the objection. SHACL does bolt closed-world cardinality onto the graph — but only over nodes its targets actually reach. A record that fails to declare its type is not rejected; it is silently *not validated*. The fail-open moves from the constraint to the targeting, which is harder to notice and harder to test.
- *"It's all one model, so drift is impossible."* — Drift is not impossible; it is merely invisible, because there is nothing to compare against.

**The deployment-shaped version of the objection:** a reasoner or shape validator on a request path is a runtime dependency with unbounded evaluation cost, whose failure mode is either blocking the system or being bypassed. Neither is acceptable on a decision path.

### The correct layering

| Concern | Owner |
|---|---|
| Rejecting a malformed record | Contracts — fail-closed parse |
| Preventing absence from becoming zero | Contracts — no defaults, explicit validity |
| Enforcing who may write what | Contracts — authority-scoped writers |
| Guaranteeing a consumer cannot see a field | Contracts — structural blinding (the field is absent from the view type) |
| Deciding adopt / reject | Application code, against a **versioned policy record** |
| Recording what was decided, by whom, from what | Ontology / provenance graph |
| Answering "what did this descend from?" | Ontology |
| Interoperating with an external group | Ontology |
| Detecting that emitted records violate a shape | Ontology — **in CI**, after the fact |

**Declarative policy is a good idea; SHACL-on-the-request-path is not the way to get it.** Put thresholds in a versioned policy record that the decision code reads, and have every decision bind the policy version in force. That gives auditability and re-derivability without putting a graph engine in the control path — see `/contract-engineering` (`versioned-policy-parameters`).

### What to say when asked to invert it

Not "no." Give them what they actually want:

1. **Auditable, declarative policy** → versioned policy records + decisions binding their policy version. Better than SHACL for this, because it re-derives historical decisions exactly.
2. **One model, not two** → generate the JSON-LD context and the shapes **from** the contract definitions. One authority, two artifacts, no drift by construction.
3. **A queryable authoritative record** → the graph, rebuilt from the contract-layer event log, and demonstrably reproducible from it.

That is the same deadline, the same benefits, and no inversion.

---

## The sync check — what makes Law 2 real

**A law with no test is a slogan.** Stated and not enforced, the projection law fails in the least visible way available: the ontology quietly stops describing the contracts, and becomes a second, wrong source of truth that nobody notices because nothing compares them.

Every ontology term that claims to describe a contract field carries a machine-checkable link to that field. CI asserts:

```
1. Every contract field referenced by the ontology still exists,
   with the type and cardinality the ontology asserts.
2. Every ontology term referenced by a contract annotation still resolves.
3. Every contract field that MUST be described (measurements, decisions,
   identities, arm roles) has a mapping — no silent omissions.
4. No shape evaluation or reasoner invocation appears on a runtime path.
5. Contract schema version and mapping version are compatible;
   a contract major version bump fails the check until the mapping is updated.

On any disagreement: the contract is correct. The ontology is corrected or
the term is deprecated. Never the reverse.
```

Checks 1 and 2 are mechanical — walk the mapping, resolve both ends. Check 3 needs a declared must-describe list. Check 4 is a lint over the production dependency graph for named reasoner and shape-engine libraries (pyshacl, owlrl, HermiT, RDFLib's inference modules), asserting zero imports outside CI and offline analysis — extend the list whenever a wrapper turns up. Check 5 is the one that catches slow rot.

**Failing this check blocks a merge. It does not block a production run.** That distinction is the architecture in one sentence.

---

## Red flags

Any of these means the layering has inverted or is about to. Stop.

| Signal | What it means |
|---|---|
| "Retire the typed schemas" | The contracts are being replaced by something with open-world semantics. |
| A reasoner or SHACL call in a request handler | The description layer is now a runtime dependency. |
| The graph is written directly, not derived from records | Two writers, no reconciliation. |
| A decision's authoritative outcome is stored only in the graph | The graph has become the record of authority. |
| "The ontology says X but the record says Y — update the record" | Law 2 inverted, explicitly. |
| Ontology and contracts maintained by different teams with no sync check | Drift with nobody assigned to notice. |
| The formalisation is required for the system to *run* | It was supposed to describe the system, not be it. |

**The test:** *turn the graph off entirely. Does the system still run correctly, and do the records still enforce everything they enforced before?* If no, the layering is wrong. If yes, the formalisation is doing its job — describing a system that stands on its own.
