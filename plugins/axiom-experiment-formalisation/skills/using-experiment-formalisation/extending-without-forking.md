# Extending Without Forking

**EXPO will run out. That is expected — it is a skeleton, deliberately domain-neutral, and your domain is not neutral.** Extension is the normal case, not a failure. What separates a survivable extension from a liability is a small set of decisions made deliberately rather than by accident.

## The mint ladder

Work down. Stop at the first rung that holds.

| Rung | Action | When |
|---|---|---|
| **1. Reuse** | Use the existing term as-is. | An existing term means what you mean. Check EXPO, then PROV-O, then your upper ontology, then domain ontologies (SIO is a good gap-filler), then [`prior-art-map.md`](prior-art-map.md). |
| **2. Narrow** | Subclass an existing term. | Yours is a *kind of* the existing thing, and every instance of yours is an instance of theirs. Test that sentence literally. |
| **3. Annotate** | Keep the existing class; add a property or attribute. | The difference is a *property* of the thing, not a different kind of thing. Status, mode, role, and phase almost always belong here. |
| **4. Mint** | Create a new term in your namespace. | Nothing above fits. Record *why* in the gap register. |

**Rung 3 is the one people skip**, and skipping it is the single largest cause of class explosion. `FailedRun`, `BlindedDecision`, `WithdrawnScaffold` are attributes wearing class costumes. The giveaway: if an instance can *stop* being one, it is not a subclass. (Strictly, OWL has no temporal semantics and will not stop you; the principle is rigidity — a class you can leave was never the right class.)

**Before minting, ask the reuse question honestly:** *"Has anyone modelling experiments needed this?"* Replication, lineage, roles, units, absence — all have prior art. Genuinely novel concepts exist (staged scaffold withdrawal, influence coefficients on partial interventions), but they are rarer than the mint rate in most projects suggests.

## Why a wrong IRI is silent — the mechanism behind Law 1

**An unresolvable or misspelled IRI in RDF does not error.** It is simply a URI that nothing else uses: it matches nothing, joins to nothing, entails nothing. The graph loads. The shapes pass. Queries return zero rows and everyone reads that as "no data yet."

The result is an ontology that looks complete and means nothing — invisibly, for as long as nobody queries across it. This is why a citation to a class that does not exist is not a typo but a defect class, and why "it parsed" is worthless as evidence.

**Convert the uncertainty into tooling.** Pin the ontologies you align to by checksum, and add a CI step that resolves **every** external IRI you reference against the pinned files, failing the build on any that do not. Then a wrong name breaks the build on day one instead of producing quiet garbage for a year.

When a check fails, classify it: wrong namespace, wrong spelling, or genuinely absent. **Never invent a term to make the check green** — record the gap and mint in your own namespace.

## Namespaces and IRIs

**Mint in your own namespace. Always.** Never assert a term into someone else's IRI space, and never assert `owl:equivalentClass` against an IRI you cannot resolve — that includes EXPO's, whose namespace IRI does not dereference and whose host you do not control ([`expo-verified-inventory.md`](expo-verified-inventory.md)). Note these are different problems: the *artifact* is obtainable and hash-verifiable; the *IRI* is not dereferenceable.

IRI rules that save pain later:

- **Opaque or semantic, but decide once.** Semantic IRIs (`.../ExperimentalArm`) are readable and become lies when meaning drifts. Opaque IRIs (`.../t_00417`) never lie and are unreadable. Semantic is usually right for a small extension; whichever you choose, do not mix.
- **No hostnames you do not control.** An IRI is an identifier, but it will be dereferenced by someone eventually. Pointing at a host you will lose is a slow-motion outage.
- **No environment, no version, no run id in the IRI.** `.../v2/Arm` and `.../staging/Arm` fragment identity permanently. Versioning belongs in `owl:versionIRI`, not the term IRI.
- **A term IRI is forever.** Once published, it is never reused for a different meaning. Deprecate and mint a new one instead.

## Relate to what you extend — in prose, and precisely

For each minted term, record:

- what it means, in one sentence a domain expert would accept;
- the competency question(s) that justify it ([`competency-questions-first.md`](competency-questions-first.md));
- the nearest existing term, and **why it did not fit**;
- the relationship you *are* asserting (`rdfs:subClassOf`, nothing, or a documented informal correspondence).

That fourth item is where discipline shows. "Close to EXPO's `ComparisonControl_TargetGroups`, but that subtree assumes group comparison and ours is per-trajectory" is a real record. `owl:equivalentClass` against an unresolvable IRI is not.

**The `experimentalControl` case is the canonical worked example** ([`sumo-upper-binding.md`](sumo-upper-binding.md)): MILO's predicate constrains argument 2 to `Object`, your control arm is a process, so you mint a sibling and document the divergence rather than asserting a subproperty that violates the domain.

## Keep the relation set small

EXPO used essentially five relations at the design level — `subclass`, `instance-of`, `part-of`, `attribute-of`, plus a role predicate described in its documentation. (The shipped OWL declares 78 object properties, almost all `has_*` renderings of attribute-of, and no role property at all — so grep the artifact before quoting the number.) Copy the restraint, not the count.

**Relations proliferate faster than classes and are far harder to retire**, because every one becomes a query someone wrote. Before minting a relation, check whether an existing one plus an attribute expresses the same thing. `hadRole` on a qualified association ([`provenance-and-lineage.md`](provenance-and-lineage.md)) replaces an entire family of `wasVerifiedBy` / `wasJudgedBy` / `wasCommissionedBy` predicates, and stays correct when a new role appears.

## Modularity

Split your extension into modules with one job each, and let dependencies run one way only:

```
your-core        (identity, records, absence encoding)
     ▲
     ├── your-comparison   (arms, factors, units of analysis)
     ├── your-governance   (roles, authorisations)
     └── your-lifecycle    (states, transitions, influence)
```

Reasons this pays: modules can be versioned independently; a consumer can import only what it needs; and a module nobody imports is visibly dead rather than invisibly carried. The one-way rule matters — a cycle between modules means they are one module that has not admitted it.

**Follow the OBO Foundry principles even if you never touch an OBO ontology.** Open, common format, unique IRI space, versioned, documented, clearly bounded, maintained. It is the best short statement of what makes a domain extension survivable, and it costs nothing to comply with from the start.

## Versioning

Version the **module**, not the terms. Use `owl:versionIRI` (or your equivalent) and a changelog.

| Change | Version impact |
|---|---|
| Add a term | Minor. Additive, no consumer breaks. |
| Add a property to an existing term | Minor, **if** it is optional. |
| Tighten a constraint (new required field, narrower domain) | **Major.** Records that were valid are now invalid. |
| Change what a term *means* while keeping its IRI | **Never do this.** Deprecate and mint. |
| Deprecate a term | Minor to announce, major to remove. |
| Change upper ontology | Major, and treat it as a migration project. |

**The unforgivable one is row 3 read together with row 5.** Silently redefining a term makes every historical record ambiguous forever, and unlike a schema change there is no error to notice. This is the same discipline as contract versioning — if you are already running `/contract-engineering`, apply that pack's rules here rather than inventing a second scheme.

**Deprecate properly:** mark the term deprecated, state the replacement, keep it resolvable, and give consumers a window. A term that vanishes breaks graphs you cannot see.

## The gap register

The artifact that makes extension reviewable. One row per gap, maintained alongside the mapping:

| Gap | Why nothing existing fits | Competency question | Resolution | Module | Status |
|---|---|---|---|---|---|
| Replication kinds | EXPO has no replication term; PROV-O models derivation, not experimental repetition | CQ-2 | mint 3 terms | comparison | minted |
| Control over a process | MILO `experimentalControl` domain 2 = `Object` | CQ-1 | mint sibling predicate | comparison | minted |
| Delegated authorisation | Absent from EXPO and SUMO | CQ-3 | mint | governance | minted |
| Influence coefficient | Absent; no prior art found | CQ-7 | mint | lifecycle | minted |

**Read the register as a diagnostic.** A short register against a large extension means terms were minted without checking for prior art. A register where every row says "nothing existing fits" and none cites a specific constraint means the checking was nominal. And a register that grows every sprint is telling you the grammar was not stable enough to formalise — which is the premature-generalisation failure, and the honest response is to stop and revisit [`formalisation-triage.md`](formalisation-triage.md).

## Checklist

- [ ] Mint ladder walked in order; each mint records what it rejected and why.
- [ ] Every term traces to a competency question.
- [ ] Own namespace; no assertions into anyone else's IRI space.
- [ ] No `owl:equivalentClass` to an unresolvable IRI.
- [ ] No version, environment, or run identifier inside a term IRI.
- [ ] Relation count kept near EXPO's restraint; roles modelled as relations, not classes.
- [ ] Attributes used for things that can change; subclasses only for things that cannot.
- [ ] Modules with one job each and acyclic dependencies.
- [ ] `owl:versionIRI` plus a changelog; meaning changes always mint, never redefine.
- [ ] Gap register maintained and reviewed for the three diagnostics above.
