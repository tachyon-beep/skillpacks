# Governance and Role Extensions

**EXPO models the experiment. It does not model who was allowed to do what.** For an experiment run by one lab that is fine. For an apparatus where different components observe, commission, design, verify, and judge — and where the point of separating them is that no single component can manufacture a favourable result — the authority structure *is* part of the experimental design, and a formalisation that omits it cannot support the claims the design was built to support.

This is the largest genuine gap in EXPO, and it is a legitimate mint. This sheet gives the pattern.

## Why authority separation is an experimental concern

Separation of powers in an experimental apparatus is not bureaucracy — it is bias control, and bias control is experimental design.

- If the component that *proposes* an intervention also *evaluates* it, the evaluation is not independent evidence.
- If the component that *observes* also *interprets*, downstream consumers receive an opinion labelled as a measurement, and cannot separate the two.
- If the component that *certifies evidence* also *decides*, "the evidence supported adoption" becomes unfalsifiable.

EXPO already recognises the underlying phenomenon: `ExperimenterBias` and `TestingEffect` sit under `AlternativeHypothesis` — rival explanations of your result. **Authority separation is the structural control for exactly those alternatives, and the record should say which separations were in force.**

## Roles are relations, not classes

The first modelling decision, and the one most often got wrong.

**Wrong:** `Verifier` as a subclass of `Agent`. It breaks immediately — the same component verifies in one activity and is verified in another, and subclass membership cannot change with context.

**Right:** a role is something an agent *plays in a specific activity*. Three ways to say it, all equivalent in spirit:

1. **PROV-O** — `qualifiedAssociation` → `Association` with `prov:agent` and `prov:hadRole`. Reuse this if you are recording provenance at all; it is free and standard.
2. **EXPO's `Role` classes** — EXPO's documentation describes a `role(?OBJECT, ?ENTITY)` predicate (its gloss: an object such as a human can play the role of subject-of-experiment or of object-of-experiment), but **the shipped OWL has no role object property at all** — zero of its 78 — only a `Role` class hierarchy (`ActorRole`, `ProductRole`, `Process-relatedRole`, `TaskRole`, `SubjectRole`). There is nothing here to reuse as a relation; one more reason to take PROV-O's.
3. **Your own ternary** — (agent, activity, role) as a first-class record.

**Use PROV-O's.** Reinventing role assertion is the reinvention failure mode, and `hadRole` also collapses a whole family of would-be predicates (`wasVerifiedBy`, `wasJudgedBy`, `wasCommissionedBy`) into one that stays correct when a new role appears ([`extending-without-forking.md`](extending-without-forking.md)).

## The role archetypes

A vocabulary of roles that recurs across accountable experimental apparatus. Use it as a checklist, not a mandate — take the ones your system actually separates, name them in your own terms, and *say which separations are load-bearing*.

| Archetype | Does | Must not | Rival explanation it controls |
|---|---|---|---|
| **Observer** | Emits measurements | Interpret, diagnose, or annotate with conclusions | Observation contaminated by expectation |
| **Commissioner** | Decides work happens; sets scope, budget, deadline | Specify the solution or supply a diagnosis | Commissioner steering the answer |
| **Designer** | Produces the candidate intervention | Evaluate its own candidate | Self-evaluation |
| **Conformance checker** | Rejects structurally invalid candidates | Judge value or merit | Structural gate becoming a quality gate |
| **Transformer** | Compiles/prepares for execution | Change meaning | Tested object ≠ deployed object |
| **Verifier** | Certifies what the artifact does; produces evidence | Decide adoption; know the candidate's origin | Evidence shaped to a desired verdict |
| **Judge** | Applies policy to evidence; issues the decision | Produce evidence; run tests | Judge manufacturing its own support |
| **Integrator** | Embodies the decision | Act without an authorisation | Unwarranted change |
| **Archivist** | Retains everything, including failures | Decide what is worth keeping | Survivorship bias |

**The "must not" column is the valuable one.** A role vocabulary without prohibitions is a labelling scheme. The prohibitions are what make the separation checkable — and what make a violation a *finding* rather than a matter of taste.

## Modelling a prohibition

You cannot enforce a prohibition in the ontology, but you can make its violation **queryable**, which is the point.

Three assertions per load-bearing separation:

1. **The separation, declared** — a record stating that roles R1 and R2 must not be played by the same agent within scope S. This is a first-class, versioned statement, not a comment.
2. **The role assignments, recorded** — every activity carries its agent-role associations.
3. **The check, written** — a query returning every activity (or scope) where the declared separation is violated. It must return zero rows.

**The query is only meaningful over canonical agent identities.** One component known under two IRIs — a redeploy, a rename, a per-environment identifier — plays both roles while the co-occurrence query passes trivially. Resolve aliases first, or the check is vacuous in exactly the way [`validation-and-conformance.md`](validation-and-conformance.md) catalogues. The identity discipline is in `/contract-engineering`, which this sheet already sends you to.

This turns "the verifier must not judge" from documentation into a test. And it survives reorganisation: when someone wires a component into a second role, the query fails rather than the invariant quietly lapsing.

**What the ontology cannot do** is stop the component from acting. That is the contract layer's job — structural blinding, authority-scoped writers, one authority per record class. Load `/contract-engineering`. The division is the projection law: **the graph reports the violation; the contracts prevent it.**

## Non-editorialising observation

A recurring and under-modelled requirement: the observer publishes measurements, and downstream consumers must be able to tell measurement from interpretation.

Model it as **provenance plus a typed distinction**, not as a promise:

- Observation records carry the observer's identity, the instrument, the time, and the raw measured values with units and validity status.
- Interpretations are **separate entities** that `wasDerivedFrom` observations and are attributed to the interpreting agent.
- A consumer can therefore ask: *"give me the observations, excluding anything derived."*

If interpretation and observation live in the same record with no marker, that query is impossible and the separation is nominal. The checkable form, stated so it can actually be queried against the structure above: **no entity typed as an observation is attributed to an agent playing any role other than observer in its generating activity, and no observation `wasDerivedFrom` another entity.** (Per-property producing-agent metadata does not exist in this model, so do not write the invariant in terms of it.)

## Delegated authorisation

The concept behind words like *warrant*, *permit*, *approval token*: **a decision that licenses a specific future action, within limits, revocably.**

**Name the near-misses before minting** — both ship, and claiming the concept is simply absent is the Law 1 failure in reverse:

- **EXPO** has `PermissionStatus` (< `StatusExperimentalDocument`). That is *document-access* status — who may read the record — not authority to act. Wrong concept, similar word.
- **SUMO** genuinely names the phenomenon: `Permission` is a `DeonticAttribute` ("permitted, by some authority, to make true"), alongside `Obligation` and `Prohibition`, plus `holdsRight` and `confersRight` — the latter documented as an entity authorising a cognitive agent to bring it about that a formula is true. That is delegated authorisation, by name.

**Mint anyway, but for the right reason.** SUMO attaches modality to *formulas*; it supplies no record structure — no issuer, scope, bounds, validity window, revocation, or content-addressed subject. You need a record, not a modal operator. If you are bound to SUMO, anchor your mint beside the deontic layer rather than beside nothing, and say in the gap register that the gap is structural rather than conceptual.

Model it as a first-class record, not a boolean on a decision:

| Field | Why |
|---|---|
| **Issuer** | The agent in the judge role. |
| **Subject** | What is authorised — the specific artifact, by content-addressed identity. |
| **Scope** | Which actions, in which region, under which regime. |
| **Bounds** | Limits: maximum influence, budget, duration. |
| **Validity window** | When it takes effect and when it expires. |
| **Evidence** | The evidence records the decision consumed. |
| **Policy version** | The thresholds in force. Without this the decision cannot be re-derived. |
| **Revocation** | Whether, when, by whom, and why. |

**Content-addressed subject identity is the load-bearing field.** An authorisation naming "candidate 47" authorises whatever candidate 47 later becomes; one naming a hash authorises exactly what was tested. This is where the formalisation earns its keep — it makes *tested object ≠ integrated object* a detectable condition rather than an incident.

**Authorisations expire and are revoked.** A record with no expiry models permission as permanent, which no real system means. And revocation is an *event*, not a field edit — see [`lifecycle-and-staged-protocol-extensions.md`](lifecycle-and-staged-protocol-extensions.md).

## Decisions bind their policy

A decision record that does not name the policy version in force is uninterpretable a month later, because the thresholds moved. Every decision binds: the evidence it consumed, the policy version, the outcome, and — where the design demands it — which view of the evidence it saw (blinded or not).

**The outcome vocabulary must be able to express refusal.** If the enum is `{adopt, defer}`, the apparatus cannot record that it rejected something, and the archive will show a perfect record forever. Include reject. Include "the no-intervention alternative won."

## Competency questions

1. Which agent played which role in this activity?
2. Has any agent played two roles declared incompatible, anywhere in scope?
3. For this decision: what evidence, which policy version, which view of the evidence?
4. Which authorisation permitted this integration, and was it valid and unrevoked at the time?
5. Does the artifact that was integrated have the same content identity as the one that was verified?
6. Which records are observations, and which are interpretations derived from them?
7. Which authorisations were revoked, when, and why?

**Question 5 is the one that finds real incidents.**

## Checklist

- [ ] Roles modelled as relations (PROV-O `hadRole`), never as agent subclasses.
- [ ] Role archetypes mapped to your components, in your own vocabulary.
- [ ] Load-bearing separations declared as versioned records, not prose.
- [ ] A zero-row query per declared separation.
- [ ] Observations and interpretations are distinct entities with derivation links.
- [ ] Authorisation is a first-class record with issuer, content-addressed subject, scope, bounds, window, evidence, policy version, revocation.
- [ ] Decisions bind evidence, policy version, and view.
- [ ] Outcome vocabulary can express reject and no-intervention-won.
- [ ] Enforcement located in the contract layer; the graph reports, never blocks.
