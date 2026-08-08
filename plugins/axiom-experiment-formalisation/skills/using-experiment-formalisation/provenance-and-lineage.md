# Provenance and Lineage

**Do not invent a lineage vocabulary. PROV-O exists, it is a W3C Recommendation, it is stable, and it is the single most reusable thing in this entire landscape.**

EXPO has no provenance layer — no `ExperimentRecord`, no `Lineage`, whatever mapping tables claim. That is not a defect; it is a boundary. EXPO models the *experiment*; PROV-O models *who produced what, from what, when*. You need both, and they compose cleanly.

## PROV-O in one screen

Three core classes and the relations between them:

```
Entity  ──wasGeneratedBy──▶  Activity  ──wasAssociatedWith──▶  Agent
  │                             │                                 │
  │◀────────used────────────────┘                                 │
  │                                                               │
  └──wasDerivedFrom──▶ Entity        wasAttributedTo ─────────────┘

Activity ──wasInformedBy──▶ Activity     (one activity used another's output)
Entity   ──wasRevisionOf──▶  Entity      (a specialisation of wasDerivedFrom)
```

| PROV-O term | In an experiment |
|---|---|
| `Entity` | A snapshot, a candidate artifact, a result set, a decision record, a dataset, a config. |
| `Activity` | A run, a compilation, a QA execution, an adjudication, a training branch. |
| `Agent` | The component, service, person, or model that acted. `SoftwareAgent` is a subclass — use it. |
| `used` | Activity consumed this entity. |
| `wasGeneratedBy` | Entity was produced by this activity. |
| `wasDerivedFrom` | Entity descended from that entity. |
| `wasAttributedTo` | Entity is credited to this agent. |
| `wasAssociatedWith` | Activity was carried out by this agent. |
| `qualifiedAssociation` + `hadRole` | **The role an agent played in a specific activity.** |

**`hadRole` is the term to notice.** It is how you say "this agent acted as the verifier in *this* activity" without making `Verifier` a class — the same role discipline EXPO reaches for with its `Role` predicate. See [`governance-role-extensions.md`](governance-role-extensions.md).

## Plan versus execution

Every experiment has two objects that novices merge: **the protocol you intended** and **the run you got**. Merging them makes it impossible to ask whether the run followed the protocol — which is the question audits actually ask.

Four ways to keep them apart, in order of cost:

1. **PROV-O itself** — `prov:Plan` is an `Entity`, attached to an activity's `prov:qualifiedAssociation` via `prov:hadPlan`. **Start here.** It is free if you are already qualifying associations, which this sheet's checklist has you doing anyway, and it is native to the vocabulary this sheet mandates. Reaching past it for SUMO or a mint is the reinvention failure in miniature.
2. **SUMO** — a `Plan` (`Procedure` → `Proposition` → `Abstract`) related to the `Process` by `realization`. Free if you have already bound to SUMO; note the signature is `(realization ?PROCESS ?PROP)` and the semantics are representational.
3. **P-Plan** — a PROV-O extension modelling plan *steps* and variables against the activities that realised them. Dormant, but the pattern is worth copying when you need step-level structure that `prov:hadPlan` alone does not give.
4. **Your own two record types** — a protocol record with an identity and a version, and a run record carrying that identity. Often sufficient, and it is Tier 1.

Whichever you pick, the record must support: *"show me the runs that claim protocol P v3, and the ones whose actual actions departed from it."* If it cannot, the protocol is decoration.

## What to record at generation time — not after

**Provenance reconstructed later is not provenance; it is a hypothesis about the past.** Log-scraping produces a lineage graph that is plausible, unfalsifiable, and wrong in exactly the cases you built it for.

The rule: **an entity's provenance is written by the activity that generated it, in the same transaction that generates it.** If the generating step cannot name its inputs, it should fail rather than emit an entity with unknown ancestry.

Minimum viable record per entity:

- a stable **identity** (content-addressed where meaningful — see `/contract-engineering`'s canonical identity discipline),
- the **activity** that generated it,
- the **entities that activity used**,
- the **agent** and its **role** in that activity,
- the **time**,
- the **version of the code or policy** in force.

That last one is routinely dropped and routinely needed. A decision record without the version of the thresholds in force cannot be re-derived, which means it cannot be audited, only believed.

## Retention: failures are the point

EXPO's stance, and FAIR's, and every serious archival standard's: **the record includes the null results, the failures, and the runs you would rather forget.**

Retain, as first-class entities:

- proposals rejected before execution,
- runs that crashed or were abandoned (with the reason),
- results that failed quality checks,
- decisions to reject or defer, with their evidence,
- **comparisons where the no-intervention arm won.**

A corpus that contains only successes cannot support any claim about rate of success — the denominator has been deleted. It also cannot support learning from failure, which is usually the more valuable signal.

**Make this checkable:** a competency question of the form *"what fraction of generated candidates were rejected, by stage?"* should return a real distribution. If it returns nothing, or returns 100% success, the retention layer is broken regardless of what the policy document says.

## Packaging: RO-Crate

When an experiment record leaves the building — archived, published, handed to a collaborator, submitted with a paper — it needs to be self-describing.

**RO-Crate** is the mature answer: a directory plus an inline JSON-LD manifest describing what is inside and how the parts relate, mostly in schema.org terms. Version 1.2 was released June 2025 and is declared stable; work toward RO-Crate 2 (modularisation) is in progress.

Use it when anything leaves the building. Skip it when everything stays in one system and you need query rather than transport. It is a *container* convention, not an ontology — it complements your PROV-O graph rather than replacing it.

## Where provenance stops and the contract layer starts

The projection law applies here with full force ([`the-projection-law.md`](the-projection-law.md)):

| Concern | Owner |
|---|---|
| Recording what happened, queryably | The provenance graph |
| Making it *impossible* to emit an entity with unknown ancestry | The contract layer (required field, fail-closed parse) |
| Answering "what did this result descend from?" | The provenance graph |
| Refusing to run a stage whose inputs are unverified | The contract layer |
| Proving an artifact is byte-identical to the one tested | Content-addressed identity in the contracts; the graph records the hash |

**The graph is where you look things up. It is never where enforcement happens.** A SHACL shape that fails on missing provenance is a useful CI check on your records; it is not a runtime gate, and the moment it becomes one you have inverted the layering.

For whole-system replay guarantees — deterministic re-execution, divergence localisation, snapshot completeness — load `/determinism-and-replay`. Provenance tells you what happened; replay tells you it would happen again.

## Competency questions

1. What did this result descend from, transitively, back to the initial state?
2. Which agent generated this artifact, in which role, under which code version?
3. Which activities used this input, and did any of them depart from the protocol they claim?
4. What fraction of candidates were rejected at each stage?
5. Show every retained run in which the no-intervention arm won.
6. For this decision, which policy version was in force, and what evidence did it consume?

## Checklist

- [ ] PROV-O reused directly for lineage; no hand-rolled `Lineage`/`ExperimentRecord` vocabulary.
- [ ] Agent roles expressed via `qualifiedAssociation` + `hadRole`, not as agent subclasses.
- [ ] Plan and execution are separate entities with an explicit realisation link.
- [ ] Provenance written by the generating activity, never reconstructed from logs.
- [ ] Code/policy version recorded on every generated entity.
- [ ] Failures, rejections, and null results retained as first-class entities.
- [ ] Retention verified by a query that returns a real distribution, not by policy assertion.
- [ ] RO-Crate (or equivalent) used for anything that leaves the system.
- [ ] No SHACL shape or reasoner sits on a runtime path.
