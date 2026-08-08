# Measurement, Uncertainty, and Absence

**Three rules, in priority order:**

1. **A measurement is a quantity with a unit, never a bare number.**
2. **A measurement without an uncertainty is a claim you cannot defend.**
3. **Unmeasured is not zero, and the record must be structurally incapable of confusing them.**

Rule 3 is a formalisation invariant in this pack — the critic checks it, and a formalisation that violates it is reported as incomplete regardless of how good the rest is.

## What EXPO gives you

EXPO models the *error* side well and the *measurement* side thinly:

Verified against the shipped OWL:

| Concept | EXPO fragment | Note |
|---|---|---|
| An observation | `ExperimentalObservation` < `ProductRole` < `Process-relatedRole` < `Role` | **Present** — and it is a *role*, not an object. |
| The results container | `ExperimentalResults` < `ProductRole` | Plural. The singular never shipped. |
| Error (general) | `Error` < `ProductRole` | Root of the measurement-error subtree, but not the direct parent of most of it: `Error` > `ObservationalError` > `MeasurementError` > {`RandomError` > `StatisticalError`, `SystematicError`, `CoarseError`}. |
| Error in a result | `ResultError` < `Error` | Parent of `IncompleteDataError`, `ErrorOfConclusion`, `FaultyComparison`, `HypothesisAcceptanceMistake`. |
| Wrong conclusion | `ErrorOfConclusion` | |
| Invalid comparison | `FaultyComparison` | First-class — see [`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md). |
| Missing data | `IncompleteDataError` | **EXPO's absence hook. Use it.** |
| False positive / negative | `FalsePpositive` *(sic)*, `FalseNegative` | Under `HypothesisAcceptanceMistake`. The misspelling is in the shipped OWL — reproduce it. |
| Significance threshold | `LevelOfSignificance` < `StatisticsCharacteristic` | |
| A measured quantity | `PhysicalQuantity` | Reused from SUMO. |

There is **no EXPO class called `Measurement`** and no `ResultSet`. SUMO supplies `Measuring` (a `Calculating` process) and the `measure` predicate, plus `PhysicalQuantity` → `UnitOfMeasure`.

**Note the modelling consequence of the first two rows:** observations, results, and errors are all *roles played by data in a process*, sitting under `Abstract`, not stored artifacts under `Physical`. If your schema treats a result as a file, resolve that mismatch deliberately rather than discovering it after the mapping ships.

**`IncompleteDataError` is the term most worth knowing here.** EXPO anticipated that missingness is a property of the *result*, not a gap in a table — and gave it a typed home in the error subtree rather than leaving it to a null.

## Units

| Approach | When |
|---|---|
| **UCUM codes in a typed field** | Default. A compact, parseable string (`mg/dL`, `ms`, `1`) alongside the value. Cheap, ubiquitous, sufficient for most projects. |
| **QUDT** | You need units as first-class graph entities and must reason over dimensions or conversions. |
| **OM** | QUDT does not cover your quantity kinds. |

**Dimensionless is still a unit.** A ratio, a probability, a normalised score — record `1`, or a named quantity kind, rather than leaving the unit field empty. An empty unit field is indistinguishable from a forgotten one, and that ambiguity is what unit bugs are made of.

**Record the quantity kind, not just the unit.** `ms` tells you the dimension; `wall_clock_latency` tells you what was timed. Two fields with the same unit and different quantity kinds are not comparable, and nothing in the unit alone will tell you that.

## Uncertainty

A number without an interval invites comparison it cannot support. Record, per measurement:

- the **estimate**,
- an **interval or dispersion** (CI, standard error, standard deviation, credible interval — say which),
- the **n** it was computed over,
- and **what n counts** — the unit of analysis (see the controls sheet).

That last field is the one that prevents the most damaging error in paired-branch work: `n=200` meaning 200 branches from 20 ancestor states is not `n=200` independent observations, and an interval computed as though it were will be too narrow whenever branches from a common ancestor are positively correlated — which is precisely why they were forked from one.

**Do not let the ontology layer compute statistics.** It records what was computed, by which method, over which unit. For whether the method was appropriate, load `/counterfactual-statistics`; for typed statistical-method vocabulary, see STATO in [`prior-art-map.md`](prior-art-map.md).

## Absence — the invariant

### The failure

A field could not be measured. Somewhere between producer and consumer it becomes `0`, `0.0`, `""`, `[]`, or `{}`. Downstream, `0` is a legitimate value — low latency, no errors, no drift — so a *measurement that never happened* becomes *a confident report of an excellent value*. Nothing crashes. Nothing logs. The number is simply wrong, and it is wrong in the direction that looks fine.

The ways it gets in are mundane: a schema default, a `.get(field, 0)`, a proto3 scalar whose absence is unobservable, a dataframe `fillna(0)`, a JSON-LD context that drops nulls, a SPARQL `OPTIONAL` whose unbound variable is coalesced.

### The rule

**Absence is a distinct, representable, queryable state — and "the value is absent" must be distinguishable from "the field was never part of this record."**

Encode all four states explicitly:

| State | Meaning |
|---|---|
| **Measured** | A value, with unit and uncertainty. |
| **Measured as zero** | Genuinely zero. A value like any other. |
| **Not measured** | The instrument did not report; the sample was unavailable; the probe was disabled. |
| **Not applicable** | The field has no meaning for this record kind. |

Distinguishing *not measured* from *not applicable* matters more than it looks: the first is a gap you might fill, the second is a category statement. Collapsing them produces datasets where "missingness" is uninterpretable.

### How to encode it

- **A validity mask or explicit status field** per measurement, not a sentinel value. Sentinels (`-1`, `-999`, `NaN`) get arithmetic done to them.
- **A reason for absence** where you have one — instrument fault, out of range, not yet run, disabled by configuration, redacted. Absence with a reason is data; absence without one is a hole.
- **Type the result as EXPO's `IncompleteDataError`** when incompleteness affects the result's interpretation, so it surfaces in the error subtree rather than only in a nullable column.
- **Never impute silently.** If you must impute, the imputed value carries an imputation flag and the version of the imputation policy, and the original absence remains recoverable.

### RDF and JSON-LD specifics

The graph world has its own ways to lose this:

- **A missing triple is ambiguous** — it means "unknown," "not applicable," and "we forgot" identically. If absence is meaningful, **assert it**: a triple saying the value is absent, with a reason, beats the absence of a triple.
- **JSON-LD serialisers commonly drop `null`.** Round-trip your absence encoding through your actual serialiser and assert that absence survives. This is a test, not an assumption.
- **`OPTIONAL` in SPARQL yields unbound variables.** Any `COALESCE(?v, 0)` in a query reintroduces the exact defect at read time, after you carefully avoided it at write time.
- **Open-world semantics are not on your side here.** OWL's open world means "not stated" is not "false" — which is philosophically correct and operationally useless. SHACL's closed-shape checking is what you want for "this record must carry a status for every measurement" ([`validation-and-conformance.md`](validation-and-conformance.md)).

### Where enforcement lives

The projection law: **the ontology can describe absence; only the contract layer can prevent the confusion.** A SHACL shape that flags a record with a bare `0` and no validity status is a good CI check on emitted records. It is not a runtime guard, and it runs after the damage.

The producer must be structurally unable to emit an unmarked absence — fail-closed parsing, no schema defaults on measurement fields, no tolerant readers. That is `/contract-engineering` (`silent-default-elimination`), and it is the load-bearing half. This sheet makes the record *say* what the contract layer *enforces*.

## Competency questions

1. For this measurement, what is the value, unit, quantity kind, uncertainty, and n — and what does n count?
2. Which measurements in this run were absent, and for what reason?
3. Are there records where a measurement is present with value zero and no validity status? *(should return zero rows)*
4. Which results are flagged `IncompleteDataError`, and did any decision consume them anyway?
5. Which values in this dataset were imputed, under which policy version?
6. Are any two quantities being compared that share a unit but differ in quantity kind?

**Question 3 is the invariant, expressed as a query.** Wire it into CI.

## Checklist

- [ ] Every measurement carries a unit; dimensionless is recorded as such, not left empty.
- [ ] Quantity kind recorded alongside unit.
- [ ] Uncertainty recorded with its type, its n, and what n counts.
- [ ] Four states distinguishable: measured / measured-as-zero / not-measured / not-applicable.
- [ ] Absence carries a reason where one exists.
- [ ] No sentinel values standing in for absence.
- [ ] Absence survives a real serialisation round-trip (tested, not assumed).
- [ ] No `COALESCE`/`fillna`/default in any read path over measurements.
- [ ] Imputation flagged, versioned, and reversible to the original absence.
- [ ] `IncompleteDataError` reachable and used where incompleteness affects interpretation.
- [ ] Enforcement lives in the contracts; the shapes are CI checks, not runtime gates.
