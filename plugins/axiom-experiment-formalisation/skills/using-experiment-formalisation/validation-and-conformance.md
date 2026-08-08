# Validation and Conformance

**A formalisation nobody can fail is a formalisation nobody can trust.** This sheet is about making the record layer testable — and about the specific ways ontology validation goes hollow while appearing to pass.

The layering rule applies throughout: **these are CI checks over records, never runtime gates.** A shape that blocks a production run has stopped describing and started governing, which is the inversion [`the-projection-law.md`](the-projection-law.md) forbids.

## The four things worth testing

| Test class | Question it answers | Tool |
|---|---|---|
| **Competency-question tests** | Can the formalisation answer what it was built to answer? | SPARQL, or plain functions at Tier 1 |
| **Shape conformance** | Do emitted records have the required structure? | SHACL (or JSON Schema at Tier 1) |
| **Invariant tests** | Are the pack's formalisation invariants held? | SPARQL queries asserting zero rows |
| **Sync check** | Does the ontology still describe the contracts it claims to describe? | Custom CI test — see the projection-law sheet |

**Start with the first.** Competency-question tests are the acceptance criteria; everything else is defence in depth.

## Competency-question tests

For each **Must** question ([`competency-questions-first.md`](competency-questions-first.md)):

1. The query, checked into the repository next to the ontology.
2. A **positive-control fixture** — data engineered to match. If the query returns nothing against it, the query is broken.
3. A **negative-control fixture** where meaningful — data engineered *not* to match. If the query matches it, the query is too permissive.
4. An assertion on the answer's **shape** (columns, cardinality), not just non-emptiness.

**Without the positive control, "zero rows" is indistinguishable from "query is wrong."** That confusion is how a validation suite passes for a year over a graph that silently stopped being populated.

## Shape conformance with SHACL

SHACL is the right tool for "records of this kind must have these properties, with these types and cardinalities." It is closed-world where you need it to be, which OWL is not.

What to shape:

- **Required properties** — a measurement must carry a unit and a validity status; an arm must carry an ancestor-state identifier and a role.
- **Cardinality** — a comparison has exactly one no-intervention arm; a decision has exactly one policy version.
- **Value constraints** — arm role drawn from a closed vocabulary; influence coefficient within its stated range.
- **Closed shapes** where a record must not carry unexpected properties — this is how you catch a producer quietly adding a field.

**What SHACL cannot do:** stop the producer emitting a bad record. It reports after the fact. If the constraint must never be violated, it belongs in the contract layer as a fail-closed parse, and the shape is a second net.

### OWL is not a validator

The most common mistake in this area. **OWL's open-world assumption means "not stated" is not "false."** A missing required property does not make a record invalid in OWL — it makes it *incompletely described*. A reasoner will happily accept a record with no measurements at all.

Worse, OWL will cheerfully *infer* rather than reject: declare a property's domain and a reasoner will conclude anything using it is of that type, rather than complaining that it was not. Domain and range in OWL are inference rules, not constraints. Many people have shipped an "ontology validator" that could not fail.

**Use OWL for classification and entailment. Use SHACL for validation. Do not confuse the two.**

## Reasoner checks — if you use a reasoner at all

Most projects do not need one, and should say so plainly rather than implying otherwise ([`sumo-upper-binding.md`](sumo-upper-binding.md)). If you do:

- **Consistency** — the ontology has a model. Inconsistency usually means contradictory disjointness or cardinality axioms.
- **No unsatisfiable classes** — a class that can have no instances is a modelling bug, and it is the highest-value automated check available. Wire it into CI.
- **Expected entailments** — assert the specific inferences you rely on. `subClassOf` chains you *think* hold frequently do not once disjointness is added.
- **Profile conformance** — if you claim OWL 2 EL/QL/RL, check it. A single disallowed construct drops you out of profile silently — an inverse property or a union in EL, a property chain in QL — and the fast reasoner stops applying. (Class↔individual punning is *legal* in OWL 2 DL and in the profiles; what exits DL entirely is illegal punning, such as one name used as both class and datatype, or metamodeling beyond punning.)

**Test the entailments you rely on, individually.** "The reasoner runs without error" is not a test.

## The invariants this pack asserts

Wire these as queries that must return **zero rows**. They are the formalisation invariants the `formalisation-critic` also checks by hand.

| Invariant | Query returns zero rows when… |
|---|---|
| **Verified vocabulary** | No cited external term is absent from **both** [`expo-owl-inventory.json`](expo-owl-inventory.json) and [`verified-terms.json`](verified-terms.json), and no cited term appears in `verified-terms.json`'s `do_not_cite` list. (Checking only one file fails against correct citations — most EXPO classes live in the inventory alone.) |
| **No-op present** | No comparison lacks a no-intervention arm. |
| **Pairing recorded** | No arm lacks an ancestor-state and input-sequence identifier. |
| **Unit of analysis declared** | No statistical claim lacks a declared unit. |
| **Absence not valued** | No measurement has a value with no validity status. |
| **Units present** | No measurement lacks a unit and quantity kind. |
| **Provenance complete** | No generated entity lacks a generating activity, agent role, and code/policy version. |
| **Failure retained** | No terminal stage exists with zero recorded failures. *(a heuristic — investigate rather than assume a bug)* |
| **Outcome expressible** | The outcome vocabulary includes a value meaning the no-intervention arm won. |
| **Not a runtime gate** | No shape or reasoner invocation appears on a production request path. |

The last one is checked structurally, not by query: grep the runtime for SHACL/reasoner invocations, and assert the count is zero outside CI and offline analysis.

## Vacuity: how validation goes hollow

**Every one of these passes. None of them tests anything.**

| Pattern | Why it is hollow | Fix |
|---|---|---|
| **Fixtures generated by the code under test** | The serialiser round-trips its own output. Any shared misunderstanding is invisible. | Hand-write fixtures, or capture from a different producer. |
| **Shapes derived from the data** | Generated from a sample, so the sample conforms by construction. | Derive shapes from competency questions and contracts. |
| **Every query returns zero rows and that is called a pass** | Indistinguishable from a broken query or an empty graph. | Positive controls, always. |
| **Shapes with no `sh:severity` triage** | Everything is a violation, so the report is ignored. | Separate Violation from Warning; keep Violations at zero. |
| **Only the happy path** | No fixture is *supposed* to fail. | For every shape, a fixture that violates it and is asserted to fail. |
| **A shape that matches nothing** | The vocabulary was supplied to the reasoner but never merged into the data graph — so targeting `prov:Agent` matches nothing even though the data declares `prov:SoftwareAgent` (`sh:targetClass` follows `rdfs:subClassOf` in the **data graph**, with no reasoner involved, so this is a merge failure, not an entailment failure). Or the target type is genuinely inferred rather than asserted — entailed from a domain axiom, say. Either way the shape reports success on data that plainly violates it. | Target on properties actually present (`sh:targetObjectsOf`) rather than entailed types; merge the vocabulary into the data graph; and **assert a minimum count of distinct invariants that fired** against the negative fixture, so a shape that stops matching fails the build instead of quietly passing. |
| **Unresolvable external IRIs** | A misspelled or wrong-namespace IRI does not error in RDF — it matches nothing, silently. | A CI step resolving every external IRI against checksum-pinned ontology files. |
| **Testing the ontology, not the records** | The OWL file is consistent; nothing checks what producers actually emit. | Validate real emitted records in CI, sampled from production. |
| **Assertions about future versions** | A test asserting unknown versions parse encodes fail-open behaviour. | Assert unknown versions are *rejected*. |

**The rule:** for every check, be able to name the change that would make it fail. If you cannot, the check is decoration.

## Pitfall scanning

Automated scanners (OOPS! and similar) catch the mechanical layer cheaply: missing annotations, unconnected classes, cycles, inverse-relationship errors, recursive definitions, naming inconsistency. Run one before review — it costs minutes and removes the entire class of findings a human reviewer would otherwise spend their attention on.

They will not catch: wrong modelling, hallucinated citations, missing controls, or a formalisation that answers no question. Those are the reviewer's job, and the critic agent's.

## CI wiring

A workable gate, cheapest first:

```
1. Parse            — ontology and shapes load
2. Pitfall scan     — mechanical issues
3. Unsatisfiable    — no class can be empty by construction   [if reasoning]
4. IRI resolution   — every external IRI resolves against checksum-pinned files
5. Shape conformance— emitted-record fixtures, both directions, with a minimum
                      count of distinct invariants asserted to fire on the negative
6. Invariants       — the zero-row queries above
7. Competency tests — Must questions, with controls
8. Sync check       — ontology terms ↔ contract fields still resolve
9. Layering check   — no shape/reasoner on a runtime path
```

Steps 1–3 run in seconds; step 7 is the one that catches real regressions. **Step 8 is the one people omit, and it is the one that fails silently for months** — see [`the-projection-law.md`](the-projection-law.md).

Failing this gate blocks a merge. It does not block a production run. That distinction is the whole architecture.

## Checklist

- [ ] Every Must competency question has a checked-in query, a positive control, and an answer-shape assertion.
- [ ] Negative controls where meaningful.
- [ ] SHACL used for validation; OWL not relied on to reject anything.
- [ ] Closed shapes where unexpected properties matter.
- [ ] Every shape has a fixture that violates it and is asserted to fail.
- [ ] Fixtures not generated by the producer under test.
- [ ] The nine query invariants wired as zero-row queries, plus the structural runtime-gate check.
- [ ] Unsatisfiable-class check in CI if a reasoner is used at all.
- [ ] Claimed OWL profile actually verified.
- [ ] Pitfall scanner run before human review.
- [ ] Sync check present and failing loudly on drift.
- [ ] No shape or reasoner on any runtime path — verified structurally.
