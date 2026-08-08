# Lifecycle and Staged Protocol Extensions

**EXPO models an experiment as a thing that happens once.** Many real apparatus instead have entities that persist and change — a candidate that is dormant, then active at partial strength, then fully committed, then withdrawn — and protocols that deliberately loosen over time as the system earns the right to operate under fewer constraints.

EXPO has pieces of both — `FactorLevel` and `DoseResponse` for strength, `SequentialDesign` and `TimeCourse` for staging — but nothing for lifecycle state, reversal, schedules, or withdrawal gates. Bind to what exists; mint the rest. Both mints have a standard shape.

## Lifecycle: states are attributes, transitions are events

The single most important modelling decision here, and the one most often got wrong.

**Wrong:** a class per state (`DormantCandidate`, `ActiveCandidate`, `RetiredCandidate`). Subclass membership cannot change; your candidate's state does. Everything downstream breaks the first time something transitions.

**Right:** the entity has one identity for its whole life. Its state is an **`Attribute`**. Every change is a **transition event** — a first-class record, not a field edit.

```
Entity (stable identity, content-addressed where meaningful)
  └── current_state : Attribute        ← derived, or cached from the event log
Transition (event)
  ├── subject         → the entity
  ├── from_state, to_state
  ├── at              → time
  ├── caused_by       → the activity or decision
  ├── authorised_by   → the authorisation, if the transition requires one
  └── reason          → why, in domain terms
```

**Why transitions must be events:** a mutable status field records only the present. It cannot answer "how long was it active?", "who moved it?", "did anything authorise this?", or "did it ever go backwards?" — and those are precisely the questions asked after something goes wrong. The event log is the record; the current state is a projection of it.

**Model the state machine explicitly too** — the permitted transitions as data, versioned. Then "did any transition occur that the machine does not permit?" becomes a query returning zero rows, rather than a property you hope holds.

**Reversal is a first-class transition.** Systems that model only forward progress cannot record retreat, so retreat gets recorded as something else — or not at all. If an entity can be withdrawn, wound down, or rolled back, those transitions belong in the machine with the same standing as advancement.

## Partial and reversible intervention

EXPO can already express intervention *strength*: `FactorLevel` is documented as "the various strengths, quantities, or concentration of each factor," and the `DoseResponse` strategy gives a target group a specified dose. What it cannot express is strength **changing over time and being lowered again** — no schedule, no realised trajectory, no reversibility. Real systems apply an intervention at strength (a blend coefficient, a traffic percentage, a mixing weight, a dose), raised from zero and lowerable again.

This matters formally because **an intervention at strength 0.05 is not the same intervention as at 1.0**, and a record that flattens both to "applied" cannot support any dose–response claim, nor explain why an effect appeared at one point and not another.

**Bind the strength to EXPO; mint only what is missing.** Two homes for one fact is the maintainability hazard here:

| Term | Meaning |
|---|---|
| **Influence coefficient** | The strength — bind its value to `FactorLevel`, and the design to `DoseResponse`, rather than minting a parallel strength vocabulary. Declare the scale and bounds; dimensionless is still a unit. |
| **Influence schedule** | How strength changes over time: the plan (`Abstract`) versus what actually happened (`Process`). Both. |
| **Reversibility** | Whether strength can be lowered, and by what mechanism — a gradual reduction and an abrupt cut are different interventions. |

**Record the realised trajectory, not just the intended one.** The schedule you set and the strengths actually in force during the measurement window are different facts, and only the second one supports a claim about what was measured. This is the plan-versus-execution distinction again ([`provenance-and-lineage.md`](provenance-and-lineage.md)); here it is the difference between a defensible dose–response result and an assumption.

**Measurements must be attributable to a strength.** A result recorded against "the intervention" with no coefficient in force at that moment is uninterpretable. Bind the measurement window to the trajectory.

## Staged protocol relaxation — scaffolds

Textbook experimental design, and EXPO covers half of it. When a signal is too noisy to measure, you **constrain until it is identifiable, calibrate under controlled relaxation, remove the constraint, and verify the conclusion survives.** EXPO's `SequentialDesign` (< `ExperimentalDesign`) and `TimeCourse` give you the *staging*; attach it to those. What nothing in EXPO or SUMO carries is the **withdrawal gate, the certifying authority, and the post-withdrawal role** — that is the mint.

Constraints introduced for that purpose — a simplified environment, a restricted action space, a curated set of starting examples, a high-fidelity but expensive execution mode, an extra oracle — are **scaffolds**. They are not part of the system you are trying to build; they are apparatus.

The failure they cause when unmodelled: **a result obtained under a scaffold gets reported as a result about the unscaffolded system.** The record shows an experiment and a conclusion, with nothing indicating that a support was in place. Nobody lies; the vocabulary simply had no place to put it.

### What a scaffold declaration must carry

| Field | Why |
|---|---|
| **The failure mode it protects against** | A scaffold without a named risk is an unexamined assumption. |
| **The constrained regime** | What is actually restricted, precisely. |
| **The relaxation regimes** | The intermediate steps between constrained and absent. |
| **The withdrawal gate** | A *measurable* criterion for removal. Not a date, not a judgement call. |
| **The certifying authority** | Which role may declare the gate passed. |
| **Post-withdrawal role** | What it becomes afterwards — often a permanent reference or calibration instrument. |
| **Known interactions** | Which other scaffolds it interacts with. |

### The invariants worth enforcing

1. **Every result records the scaffold state in force when it was produced.** This is the invariant. Without it, results from different regimes are silently pooled.
2. **Scaffolds withdraw independently.** One passing its gate says nothing about another. A single global "maturity level" conflates unrelated supports and is the most common shortcut here.
3. **Change one at a time**, unless the interaction *is* the declared experiment — in which case the single-withdrawal controls must already exist. Otherwise a regression cannot be attributed.
4. **Withdrawal is not deletion.** A withdrawn scaffold usually persists as a reference, calibration, or escalation path. Model the role change ([`controls-counterfactuals-and-replication.md`](controls-counterfactuals-and-replication.md)) — the artifact stays, its role transitions.

**Scaffold state is a lifecycle**, so reuse the machinery above: states (`active` → `relaxing` → `withdrawn` → `retained-as-reference`), transitions as events, gates as authorisations. It is the same pattern applied to apparatus rather than to candidates.

## Regimes and execution modes

Related and frequently conflated: a system may run in a high-fidelity, expensive mode and a cheaper approximate mode. Results from the two are **not interchangeable**, and the difference is often invisible in the numbers.

Record on every result: the **regime** it was produced under, and — where the cheaper mode is calibrated against the reference — the **measured divergence** between them. A result whose regime is unrecorded cannot be compared to one from another regime, and a formalisation that permits that comparison silently is worse than none.

Where this belongs in EXPO is arguable: `ExperimentalModel` is defined as a description of dependencies between factors and variables, so an execution/fidelity mode arguably fits `ExperimentalMethod` or `ExperimentalTechnology` (both ship) at least as naturally. Pick one, record the choice — the point is that the regime is modelled at all, not an implementation detail left off the record.

## Competency questions

1. What state is this entity in now, and what was the full transition history?
2. Did any transition occur that the state machine does not permit?
3. What authorised this transition, and was that authorisation valid at the time?
4. What influence coefficient was in force during the window this measurement covers?
5. Which scaffolds were active when this result was produced?
6. Which scaffolds have passed their withdrawal gates, certified by whom, on what measurement?
7. Were any two scaffolds withdrawn in the same interval? *(if so, attribution is compromised — flag it)*
8. Under which regime was this result produced, and what is the measured divergence from the reference regime?
9. Which entities have transitioned backwards, and why?

**Question 5 is the invariant.** If it cannot be answered for every result, scaffolded and unscaffolded results are already mixed in your corpus.

## Checklist

- [ ] Entity identity stable across its whole life; state is an attribute, not a class.
- [ ] Every state change is a transition event with cause, time, and authorisation where required.
- [ ] The permitted state machine is modelled as versioned data, with a zero-row conformance query.
- [ ] Reversal and withdrawal are first-class transitions.
- [ ] Influence coefficient recorded with scale and bounds; dimensionless recorded as such.
- [ ] Intended schedule and realised trajectory both recorded.
- [ ] Every measurement attributable to the strength in force during its window.
- [ ] Scaffolds declared with risk, regimes, measurable gate, certifying role, post-withdrawal role, interactions.
- [ ] Every result records the scaffold state in force.
- [ ] Scaffolds tracked and withdrawn independently — no single global maturity level.
- [ ] Withdrawal modelled as a role change, not a deletion.
- [ ] Every result records its execution regime, plus divergence from the reference regime where applicable.
